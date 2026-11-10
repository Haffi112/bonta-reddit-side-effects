function unwrapMarks() {
  document.querySelectorAll('[data-source-region] mark').forEach((mark) => {
    mark.replaceWith(document.createTextNode(mark.textContent || ''));
  });
  document.querySelectorAll('[data-source-region]').forEach((region) => region.normalize());
}

function highlightVerbatim(needle) {
  unwrapMarks();
  const status = document.querySelector('[data-highlight-status]');
  if (!needle) return;

  for (const region of document.querySelectorAll('[data-source-region]')) {
    const walker = document.createTreeWalker(region, NodeFilter.SHOW_TEXT);
    let node = walker.nextNode();
    while (node) {
      const text = node.nodeValue || '';
      let index = text.indexOf(needle);
      if (index < 0) index = text.toLocaleLowerCase().indexOf(needle.toLocaleLowerCase());
      if (index >= 0) {
        const before = document.createTextNode(text.slice(0, index));
        const mark = document.createElement('mark');
        mark.textContent = text.slice(index, index + needle.length);
        const after = document.createTextNode(text.slice(index + needle.length));
        node.parentNode.replaceChild(after, node);
        after.parentNode.insertBefore(mark, after);
        mark.parentNode.insertBefore(before, mark);
        mark.scrollIntoView({ behavior: 'smooth', block: 'center' });
        if (status) status.textContent = 'Exact wording highlighted in the submission.';
        return;
      }
      node = walker.nextNode();
    }
  }
  if (status) status.textContent = 'This exact wording was not found in the title or body.';
}

function mirrorMatchingDecision(input) {
  const sourceRow = input.closest('[data-field-row]');
  const sourceCard = input.closest('[data-output-card]');
  if (!sourceRow || !sourceCard) return;

  let mirrored = 0;
  let targetLabel = '';
  document.querySelectorAll('[data-field-row]').forEach((targetRow) => {
    const targetCard = targetRow.closest('[data-output-card]');
    if (!targetCard || targetCard === sourceCard) return;
    if (targetRow.dataset.fieldKey !== sourceRow.dataset.fieldKey) return;
    if (targetRow.dataset.valueSignature !== sourceRow.dataset.valueSignature) return;

    const targetInput = Array.from(targetRow.querySelectorAll('input[type="radio"]'))
      .find((candidate) => candidate.value === input.value);
    if (!targetInput || targetInput.checked) return;
    targetInput.checked = true;
    targetRow.classList.remove('auto-synced');
    window.requestAnimationFrame(() => targetRow.classList.add('auto-synced'));
    window.setTimeout(() => targetRow.classList.remove('auto-synced'), 1200);
    targetLabel = targetCard.querySelector('.response-heading h2')?.textContent || '';
    mirrored += 1;
  });

  if (mirrored) {
    const status = document.querySelector('[data-sync-status]');
    const decision = input.value === 'correct' ? 'Correct' : 'Incorrect';
    if (status) status.textContent = `${decision} also applied to ${targetLabel || 'the matching response'}.`;
  }
}

document.addEventListener('change', (event) => {
  const decision = event.target.closest('[data-field-row] input[type="radio"]');
  if (decision) mirrorMatchingDecision(decision);
});

document.addEventListener('click', (event) => {
  const locate = event.target.closest('[data-verbatim]');
  if (locate) highlightVerbatim(locate.dataset.verbatim || '');

  const markAll = event.target.closest('[data-mark-undecided]');
  if (markAll) {
    const scope = markAll.closest('[data-output-card]') || document;
    scope.querySelectorAll('[data-field-row]').forEach((row) => {
      if (!row.querySelector('input[type="radio"]:checked')) {
        const correct = row.querySelector('input[value="correct"]');
        if (correct) {
          correct.checked = true;
          mirrorMatchingDecision(correct);
        }
      }
    });
  }

  const copy = event.target.closest('[data-copy-button]');
  if (copy) {
    const value = document.querySelector('[data-copy-value]')?.textContent || '';
    navigator.clipboard.writeText(value).then(() => { copy.textContent = 'Copied'; });
  }
});
