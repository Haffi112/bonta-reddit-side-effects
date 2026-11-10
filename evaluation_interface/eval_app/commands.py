from __future__ import annotations

import subprocess
from pathlib import Path

import click
from flask import Flask
from sqlalchemy import delete, func, select

from .importer import DEFAULT_STAGES, import_evaluation_data
from .models import Annotation, ModelOutput, db


def _guard_local_private_database() -> None:
    database = db.engine.url.database
    if db.engine.url.get_backend_name() != "sqlite" or not database:
        return
    database_path = Path(database).resolve()
    if "Dropbox" not in database_path.parts:
        return
    database_path.parent.mkdir(parents=True, exist_ok=True)
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(database_path)],
        cwd=database_path.parent,
        capture_output=True,
        check=False,
    ).returncode
    if ignored != 0:
        raise click.ClickException(
            f"Refusing to import private Reddit text: {database_path} is not gitignored."
        )
    try:
        subprocess.run(
            ["xattr", "-w", "com.dropbox.ignored", "1", str(database_path.parent)],
            capture_output=True,
            text=True,
            check=True,
        )
        value = subprocess.run(
            ["xattr", "-p", "com.dropbox.ignored", str(database_path.parent)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise click.ClickException(
            f"Refusing to import private Reddit text: could not set the Dropbox exclusion on {database_path.parent}."
        ) from exc
    if value != "1":
        raise click.ClickException(
            f"Refusing to import private Reddit text: Dropbox exclusion failed for {database_path.parent}."
        )


def register_commands(app: Flask) -> None:
    @app.cli.command("init-db")
    def init_db() -> None:
        """Create database tables without changing existing data."""
        db.create_all()
        click.echo("Database tables are ready.")

    @app.cli.command("import-data")
    @click.option(
        "--source-jsonl",
        type=click.Path(path_type=Path, exists=True, dir_okay=False),
        required=True,
        help="Private filtered_dataset/submissions.jsonl file.",
    )
    @click.option(
        "--results-sqlite",
        type=click.Path(path_type=Path, exists=True, dir_okay=False),
        required=True,
        help="Private extraction checkpoint SQLite database.",
    )
    @click.option("--stage", "stages", multiple=True, default=DEFAULT_STAGES, show_default=True)
    @click.option("--limit", type=click.IntRange(min=1), default=None)
    @click.option("--dry-run", is_flag=True, help="Read and count data without writing to the app DB.")
    @click.option(
        "--confirm-private-data",
        is_flag=True,
        help="Confirm that the destination is approved to hold private Reddit text.",
    )
    def import_data(
        source_jsonl: Path,
        results_sqlite: Path,
        stages: tuple[str, ...],
        limit: int | None,
        dry_run: bool,
        confirm_private_data: bool,
    ) -> None:
        """Import the successful intersection of the selected model runs."""
        if not dry_run and not confirm_private_data:
            raise click.UsageError(
                "Refusing to copy Reddit text without --confirm-private-data. "
                "Verify that the destination database and Heroku app are private and approved."
            )
        if not dry_run:
            _guard_local_private_database()
            db.create_all()
        result = import_evaluation_data(
            source_jsonl,
            results_sqlite,
            stages=stages,
            limit=limit,
            dry_run=dry_run,
        )
        click.echo(
            f"Intersection: {result['intersection_records']:,} submissions; "
            f"{result['model_outputs']:,} blinded forms."
        )
        if not dry_run:
            click.echo(
                f"Added {result['new_submissions']:,} submissions and "
                f"{result['new_outputs']:,} outputs."
            )

    @app.cli.command("prune-excluded-models")
    @click.option(
        "--confirm",
        is_flag=True,
        help="Delete model outputs outside the configured evaluation stages.",
    )
    def prune_excluded_models(confirm: bool) -> None:
        """Remove excluded model variants, refusing to discard annotations."""
        excluded_ids = select(ModelOutput.id).where(
            ModelOutput.model_key.not_in(DEFAULT_STAGES)
        )
        output_count = int(
            db.session.scalar(
                select(func.count(ModelOutput.id)).where(
                    ModelOutput.model_key.not_in(DEFAULT_STAGES)
                )
            )
            or 0
        )
        annotation_count = int(
            db.session.scalar(
                select(func.count(Annotation.id)).where(
                    Annotation.model_output_id.in_(excluded_ids)
                )
            )
            or 0
        )
        click.echo(
            f"Excluded outputs: {output_count:,}; attached annotations: {annotation_count:,}."
        )
        if annotation_count:
            raise click.ClickException(
                "Refusing to delete excluded outputs that already have annotations."
            )
        if not confirm:
            click.echo("No rows deleted; repeat with --confirm to apply the cleanup.")
            return
        db.session.execute(
            delete(ModelOutput).where(ModelOutput.model_key.not_in(DEFAULT_STAGES))
        )
        db.session.commit()
        click.echo(f"Deleted {output_count:,} excluded model outputs.")
