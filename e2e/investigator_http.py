"""Functional HTTP, PDF import and isolated browser workflows against a local server."""
import io,json,time,re
from pathlib import Path
import httpx
from reportlab.pdfgen import canvas
from playwright.sync_api import sync_playwright, expect

BASE='http://127.0.0.1:8789'

def main():
    with httpx.Client(base_url=BASE,timeout=90,trust_env=False) as client:
        b=client.get('/api/demo').json()
        assert client.post('/api/import',json=b).status_code==200
        assert client.post('/api/import',json=b,headers={'Origin':'https://evil.example'}).status_code==403
        assert client.get('/api/status',headers={'Host':'evil.example'}).status_code==400
        assert client.post('/api/cases/PAY-CREDIT/approve',json={}).status_code==422
        def oversized_chunks():
            for _ in range(6):
                yield b'x' * (1024 * 1024)
        assert client.post('/api/import',content=oversized_chunks(),headers={'Content-Type':'application/json'}).status_code==413
        assert client.post('/api/documents',data={'document_id':'unsupported'},files={'file':('x.html',b'<html>hello</html>')}).status_code==422
        pdf=io.BytesIO();c=canvas.Canvas(pdf);c.drawString(50,700,'Remittance supporting note: no ledger change requested.');c.save()
        assert client.post('/api/documents',data={'document_id':'pdf-evidence'},files={'file':('evidence.pdf',pdf.getvalue(),'application/pdf')}).status_code==200
        blank=io.BytesIO();c=canvas.Canvas(blank);c.showPage();c.save()
        assert client.post('/api/documents',data={'document_id':'blank'},files={'file':('blank.pdf',blank.getvalue(),'application/pdf')}).status_code==422
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1000})
        errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(BASE)
        expect(page.get_by_role('heading',name='Every payment. A clear explanation.')).to_be_visible()
        page.get_by_role('button',name='Load demo',exact=True).click()
        expect(page.locator('#detail h2')).to_have_text('Northstar Trading')
        page.get_by_role('button',name=re.compile('^(Investigate payment|Reinvestigate with current evidence)$')).click()
        expect(page.get_by_role('button',name='Approve this sandbox allocation')).to_be_visible(timeout=120000)
        expect(page.locator('#detail')).to_contain_text('CN-2026-04')
        with httpx.Client(base_url=BASE,timeout=90,trust_env=False) as client:
            assert client.post('/api/import',json={'documents':[{'id':f'late-note-{time.time_ns()}','text':'A new neutral review note.'}]}).status_code==200
        page.reload();page.locator('.case[data-id="PAY-CREDIT"]').click()
        expect(page.locator('#detail')).to_contain_text('New evidence has arrived since this proposal.')
        assert page.get_by_role('button',name='Approve this sandbox allocation').count()==0
        page.get_by_role('button',name='Reinvestigate with current evidence',exact=True).click()
        expect(page.get_by_role('button',name='Approve this sandbox allocation')).to_be_visible(timeout=120000)
        Path('runs').mkdir(exist_ok=True)
        page.screenshot(path='runs/payment-investigator-review.png',full_page=True)
        page.get_by_role('button',name='Approve this sandbox allocation').click()
        expect(page.locator('#detail')).to_contain_text('Reconciled in the sandbox ledger.',timeout=15000)
        assert page.get_by_role('button',name='Approve this sandbox allocation').count()==0
        page.reload();page.locator('.case[data-id="PAY-CREDIT"]').click()
        expect(page.locator('#detail')).to_contain_text('Reconciled in the sandbox ledger.')
        page.locator('.case[data-id="PAY-MISSING"]').click()
        page.get_by_role('button',name='Investigate payment',exact=True).click()
        expect(page.locator('#detail')).to_contain_text('What needs attention',timeout=120000)
        assert page.get_by_role('button',name='Approve this sandbox allocation').count()==0
        page.set_viewport_size({'width':390,'height':844})
        page.screenshot(path='runs/payment-investigator-mobile.png',full_page=True)
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        assert not errors,errors
        browser.close()
    print(json.dumps({'passed':True,'checks':['HTTP same-origin/host guards','chunked upload byte limit','bounded document type import','real PDF text extraction','image-only PDF rejection','browser demo investigation citations approval restart','stale evidence hides approval until reinvestigation','blocked case has no approve control','mobile no horizontal overflow','no browser errors']},indent=2))

if __name__=='__main__': main()
