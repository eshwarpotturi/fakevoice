const { chromium } = require('playwright');
(async () => { const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 1920, height: 1080 } });
  await p.goto('http://localhost:8766/stats.html'); await p.evaluate(() => { try { localStorage.setItem('theme','light'); } catch(e){} }); await p.reload(); await p.waitForTimeout(1500);
  await p.screenshot({ path: 'logs/story.png', fullPage: true });
  await p.goto('http://localhost:8766/index.html'); await p.waitForTimeout(2500); await p.screenshot({ path: 'logs/demo_rf.png' }); await b.close(); })();
