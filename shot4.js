const { chromium } = require('playwright');
(async () => { const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 1920, height: 1080 } });
  const logs = []; p.on('console', m => { if (m.type()==='error') logs.push(m.text().slice(0,200)); }); p.on('pageerror', e => logs.push('pageerror '+e.message.slice(0,200)));
  await p.goto('http://localhost:8766/stats.html'); await p.evaluate(() => { try { localStorage.setItem('theme','light'); } catch(e){} }); await p.reload(); await p.waitForTimeout(1500);
  const ch = await p.$$('.chapter'); console.log('chapters:', ch.length);
  await ch[ch.length-1].screenshot({ path: 'logs/chapter7.png' });
  await ch[ch.length-1].$eval('.tb', b => b.click()); await p.waitForTimeout(200);
  console.log('table rows:', (await ch[ch.length-1].$$('table tr')).length, '| console issues:', logs.length ? logs : 'none'); await b.close(); })();
