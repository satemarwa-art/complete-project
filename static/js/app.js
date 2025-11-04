document.addEventListener('DOMContentLoaded', function () {
  // Make Django forms look like Bootstrap
  document.querySelectorAll('form').forEach(function (form) {
    form.querySelectorAll('input:not([type=checkbox]):not([type=radio]):not([type=file]), textarea').forEach(function (el) {
      el.classList.add('form-control');
    });
    form.querySelectorAll('select').forEach(function (el) {
      el.classList.add('form-select');
    });
    form.querySelectorAll('input[type=file]').forEach(function (el) {
      el.classList.add('form-control');
    });
  });

  // Dropzone behavior for upload page
  const dropzone = document.getElementById('dropzone');
  if (dropzone) {
    const fileInput = dropzone.querySelector('input[type=file]');
    const prevent = (e) => { e.preventDefault(); e.stopPropagation(); };
    ['dragenter','dragover','dragleave','drop'].forEach(evt => dropzone.addEventListener(evt, prevent));
    ['dragenter','dragover'].forEach(evt => dropzone.addEventListener(evt, () => dropzone.classList.add('dragover')));
    ['dragleave','drop'].forEach(evt => dropzone.addEventListener(evt, () => dropzone.classList.remove('dragover')));
    dropzone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      if (dt && dt.files && dt.files.length) {
        fileInput.files = dt.files;
      }
    });
    dropzone.addEventListener('click', () => fileInput && fileInput.click());
  }

  // Loading state for submit buttons
  const uploadForm = document.getElementById('upload-form');
  const runForm = document.getElementById('run-form');
  const attachLoading = (formId, buttonId) => {
    const form = document.getElementById(formId);
    const btn = document.getElementById(buttonId);
    if (!form || !btn) return;
    form.addEventListener('submit', () => {
      btn.disabled = true;
      btn.querySelector('.default-text')?.classList.add('d-none');
      btn.querySelector('.loading')?.classList.remove('d-none');
    });
  };
  attachLoading('upload-form', 'submit-btn');
  attachLoading('run-form', 'run-btn');
});

