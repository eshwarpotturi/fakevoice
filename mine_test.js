const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 1920, height: 1080 } });
  await p.goto('http://localhost:8766/index.html');
  await p.waitForFunction(() => /^Ready|could not/.test(document.getElementById('status').textContent), null, { timeout: 180000 });
  console.log('buttons:', (await p.$$eval('#samples button', bs => bs.map(x => x.textContent.replace(/\s+/g,' ').trim()))).slice(0, 5).join(' | '));
  for (const phone of [false, true]) {
    await p.setChecked('#phone', phone);
    for (let i = 1; i <= 4; i++) {
      await p.click(`#samples button:nth-child(${i})`);
      await p.waitForFunction(() => /^Done|^The check|^Sorry/.test(document.getElementById('status').textContent), null, { timeout: 120000 });
      await p.waitForTimeout(100);
      console.log(phone ? 'phone' : 'clean', i, '|', (await p.textContent('#verdict')).trim(), '|', (await p.textContent('#truth')).replace(/\s+/g,' ').trim().slice(0, 95));
    }
  }
  await p.setChecked('#phone', false); await p.click('#samples button:nth-child(1)'); await p.waitForTimeout(6000); await p.screenshot({ path: 'logs/mine.png' });
  await b.close();
})();
