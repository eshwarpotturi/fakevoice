// End-to-end check of both pages in a real browser (run with: NODE_PATH=$(npm root -g) node browser_test.js)
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1920, height: 1080 } });
  const logs = [];
  p.on('console', m => { if (['error','warning'].includes(m.type())) logs.push(m.type()+': '+m.text()); });
  p.on('pageerror', e => logs.push('pageerror: '+e.message));
  await p.goto('http://localhost:8765/index.html');
  await p.waitForFunction(() => /^Ready|could not/.test(document.getElementById('status').textContent), null, { timeout: 180000 });
  console.log('status:', await p.textContent('#status'));
  const n = (await p.$$('#samples button')).length;
  let right = 0, rightPhone = 0;
  for (const phone of [false, true]) {
    await p.setChecked('#phone', phone);
    for (let i = 1; i <= n; i++) {
      await p.click(`#samples button:nth-child(${i})`);
      await p.waitForFunction(() => document.getElementById('status').textContent === 'Done.' || document.getElementById('status').textContent.startsWith('Sorry'), null, { timeout: 120000 });
      await p.waitForTimeout(50);
      const v = await p.textContent('#verdict'), t = await p.textContent('#truth');
      const ok = t.includes('detector was right'); if (ok) (phone ? rightPhone++ : right++);
      if (!ok) console.log('  miss/unsure:', phone ? '[phone]' : '[clean]', t.trim().slice(0, 100), '|', v);
    }
    if (!phone) await p.screenshot({ path: 'logs/demo.png' });
  }
  console.log(`samples right: clean ${right}/${n}, phone line ${rightPhone}/${n}`);
  await p.click('#theme'); await p.screenshot({ path: 'logs/demo_dark.png' });
  await p.goto('http://localhost:8765/stats.html'); await p.waitForTimeout(1500);
  await p.screenshot({ path: 'logs/stats.png', fullPage: true });
  await p.click('.tile >> nth=0'); await p.waitForTimeout(300); await p.screenshot({ path: 'logs/stats_popup.png' }); await p.click('#dlg-x');
  await p.click('#t1'); await p.waitForTimeout(200);
  console.log('table view rows:', (await p.$$('#chart1 tr')).length);
  await b.close();
  console.log('console issues:', logs.length ? logs.slice(0, 6) : 'none');
})();
