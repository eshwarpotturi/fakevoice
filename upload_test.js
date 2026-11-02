const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch(); const p = await b.newPage();
  const logs = []; p.on('console', m => { if (m.type()==='error') logs.push(m.text().slice(0,300)); }); p.on('pageerror', e => logs.push('pageerror '+e.message.slice(0,300)));
  await p.goto('http://localhost:8766/index.html');
  await p.waitForFunction(() => /^Ready|could not/.test(document.getElementById('status').textContent), null, {timeout:180000});
  for (const f of ['short3.wav','long21.wav','stereo48k.wav','long21.webm','long21.m4a','long21.mp4']) {
    logs.length = 0;
    await p.setInputFiles('#file', '/tmp/rt/'+f);
    await p.waitForFunction(() => /^Done|^Sorry|silent|too short/.test(document.getElementById('status').textContent), null, {timeout:120000});
    console.log(f.padEnd(14), '|', (await p.textContent('#status')).slice(0,60), '|', (await p.textContent('#verdict')).trim(), '|', logs.slice(0,2).join(' // '));
    await p.waitForTimeout(200);
  }
  await b.close();
})();
