"""Build the report PDF: Markdown -> ODT -> (patch) -> PDF.

The patch step exists because pandoc writes the table cell styles into
content.xml as automatic styles with fo:border="none", where the reference
document cannot reach them. Borders are therefore applied after conversion, on
every build, rather than by hand.

  /home/prinlab/miniconda3/envs/scte/bin/python report/build.py
"""
import os, re, shutil, subprocess, sys, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, 'SCTE_结果报告.md')
ODT = os.path.join(HERE, 'SCTE_结果报告.odt')
PDF = os.path.join(HERE, 'SCTE_结果报告.pdf')
REF = os.path.join(HERE, 'reference.odt')

RULE = '0.5pt solid #000000'          # every cell edge
HEAD_BOTTOM = '1pt solid #000000'     # heavier line under the header row
HEAD_FILL = '#ececec'


def run(cmd, **kw):
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, **kw)
    if r.returncode:
        sys.exit('失败: %s\n%s%s' % (' '.join(cmd), r.stdout[-2000:], r.stderr[-2000:]))
    return r


def patch_borders(path):
    """Give the table cells visible borders. Rewrites content.xml in place."""
    with zipfile.ZipFile(path) as z:
        items = {n: z.read(n) for n in z.namelist()}
    c = items['content.xml'].decode('utf-8')

    def cell(name, props):
        pat = (r'(<style:style style:name="%s" style:family="table-cell">\s*'
               r'<style:table-cell-properties )fo:border="none"( */>)' % name)
        new = r'\g<1>%s\g<2>' % props
        return re.subn(pat, new, c, count=1)

    n_total = 0
    c, n = cell('TableRowCell',
                'fo:border="%s" fo:padding="0.06cm 0.12cm"' % RULE)
    n_total += n
    c, n = cell('TableHeaderRowCell',
                'fo:border="%s" fo:border-bottom="%s" fo:background-color="%s" '
                'fo:padding="0.06cm 0.12cm"' % (RULE, HEAD_BOTTOM, HEAD_FILL))
    n_total += n
    if n_total != 2:
        sys.exit('单元格样式未按预期匹配（匹配到 %d/2），pandoc 输出结构可能已变' % n_total)
    items['content.xml'] = c.encode('utf-8')

    tmp = path + '.tmp'
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        # mimetype must be first and stored, or the file will not open
        z.writestr(zipfile.ZipInfo('mimetype'), items.pop('mimetype'),
                   compress_type=zipfile.ZIP_STORED)
        for n, data in items.items():
            z.writestr(n, data)
    shutil.move(tmp, path)
    return n_total


for p in (ODT, PDF):
    if os.path.exists(p):
        os.remove(p)
run(['pandoc', os.path.basename(MD), '-o', os.path.basename(ODT),
     '--reference-doc', os.path.basename(REF)])
print('已加边框的单元格样式:', patch_borders(ODT))
run(['soffice', '--headless', '--convert-to', 'pdf', '--outdir', HERE, ODT])
pages = subprocess.run(['pdfinfo', PDF], capture_output=True, text=True).stdout
print([l for l in pages.splitlines() if l.startswith('Pages')][0])
print('输出', PDF)
