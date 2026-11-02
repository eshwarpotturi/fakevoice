const { chromium } = require('playwright');
(async () => { const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 1920, height: 1080 } });
  await p.goto('http://localhost:8766/stats.html'); await p.evaluate(() => { try { localStorage.setItem('theme','light'); } catch(e){} }); await p.reload(); await p.waitForTimeout(1200);
  await p.screenshot({ path: 'logs/stats_light.png', clip: { x: 0, y: 0, width: 1920, height: 1080 } }); await b.close(); })();
