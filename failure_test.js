// Simulates the detector failing to download, then recovering when the user tries again.
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch(); const p = await b.newPage();
  let block = true;
  await p.route('**/model/detector.onnx', (r) => block ? r.abort() : r.continue());
  await p.goto('http://localhost:8766/index.html'); await p.waitForTimeout(2500);
  console.log('1 start-up   :', await p.textContent('#status'));
  await p.setInputFiles('#file', '/tmp/rt/long21.m4a');
  await p.waitForFunction(() => /^The detector could not start|^Done/.test(document.getElementById('status').textContent), null, {timeout: 60000});
  console.log('2 blocked    :', (await p.textContent('#status')).slice(0, 140));
  block = false;
  await p.setInputFiles('#file', '/tmp/rt/long21.m4a');
  await p.waitForFunction(() => /^Done|^The check/.test(document.getElementById('status').textContent), null, {timeout: 120000});
  console.log('3 after retry:', await p.textContent('#status'), '|', (await p.textContent('#verdict')).trim());
  await b.close();
})();
