import markdown, re, asyncio, base64, os
from playwright.async_api import async_playwright

md = open("/home/claude/work/paper/paper4.md").read()
# inline images as base64 so the PDF is self-contained
def inline(m):
    p = m.group(1); b = base64.b64encode(open("/home/claude/work/"+p,"rb").read()).decode()
    return f'<img src="data:image/png;base64,{b}">'
html_body = markdown.markdown(md, extensions=["tables"])
html_body = re.sub(r'<img alt="[^"]*" src="([^"]+)" ?/?>', inline, html_body)
html_body = html_body.replace("<p><img", '<p class="fig"><img')

css = """
@page { size: Letter; margin: 0.85in 0.9in 0.9in 0.9in; }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.6pt; line-height: 1.42; color: #111; }
h1 { font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 20pt; line-height: 1.15; margin: 0 0 8pt; }
h2 { font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 12.5pt; margin: 20pt 0 6pt; page-break-after: avoid; }
h3 { font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 12.5pt; font-weight: 400; color: #444; margin: 0 0 12pt; }
p { margin: 0 0 8pt; text-align: justify; }
hr { border: 0; border-top: 1px solid #ccc; margin: 10pt 0; }
table { border-collapse: collapse; width: 100%; font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 8.8pt; margin: 6pt 0 12pt; page-break-inside: avoid; }
th { text-align: left; border-bottom: 1.5px solid #333; padding: 3pt 5pt; font-weight: 600; }
td { border-bottom: 0.5px solid #ddd; padding: 3pt 5pt; vertical-align: top; }
p.fig { text-align: center; margin: 10pt 0 2pt; page-break-inside: avoid; }
p.fig img { max-width: 100%; max-height: 6.3in; }
p.fig + p { font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 8.8pt; color: #444; text-align: left; margin-bottom: 14pt; page-break-before: avoid; }
ul { margin: 0 0 8pt 16pt; padding: 0; } li { margin-bottom: 4pt; }
strong { font-weight: 700; }
em { font-style: italic; }
"""
html = f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{html_body}</body></html>"
open("/home/claude/work/paper/paper4.html","w").write(html)

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page()
        await pg.set_content(html, wait_until="load")
        await pg.pdf(path="/mnt/user-data/outputs/FixedVenues_Preprint_v2_0.pdf", format="Letter", print_background=True, display_header_footer=True,
                     header_template="<div></div>", footer_template="<div style='font-size:8px;color:#777;width:100%;text-align:center;font-family:Arial'>Fixed Venues — preprint v2.0, 19 Oct 2026 · not peer reviewed · page <span class='pageNumber'></span> of <span class='totalPages'></span></div>",
                     margin={"top":"0.85in","bottom":"0.9in","left":"0.9in","right":"0.9in"})
        await b.close()
asyncio.run(main()); print("pdf ok")

