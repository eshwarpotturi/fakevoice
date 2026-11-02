const { chromium } = require('playwright');
(async () => { const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 1920, height: 1080 } });
  const logs = []; p.on('console', m => { if (m.type()==='error') logs.push(m.text().slice(0,200)); }); p.on('pageerror', e => logs.push('pageerror '+e.message.slice(0,200)));
  await p.goto('http://localhost:8766/stats.html'); await p.evaluate(() => { try { localStorage.setItem('theme','light'); } catch(e){} }); await p.reload(); await p.waitForTimeout(1500);
  await p.screenshot({ path: 'logs/story2.png', fullPage: true });
  await p.click('.chapter:nth-of-type(2) .tb'); await p.waitForTimeout(200);
  console.log('console issues:', logs.length ? logs : 'none'); await b.close(); })();
