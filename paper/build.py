"""Build the manuscript: Markdown -> ODT -> (border patch) -> docx -> PDF.

Table borders are applied after conversion because pandoc writes the cell styles
into content.xml as automatic styles, where the reference document cannot reach
them. The docx is produced from the ODT rather than directly by pandoc, because
pandoc's default reference.docx carries Letter geometry that the layout patch
could not override.

  /home/prinlab/miniconda3/envs/scte/bin/python paper/build.py
"""
import os, re, shutil, subprocess, sys, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
MD, REF = os.path.join(HERE, 'output/doc/manuscript_npj.md'), os.path.join(HERE, 'reference.odt')
ODT, DOCX, PDF = (os.path.join(HERE, 'output/doc/manuscript.' + e) for e in ('odt', 'docx', 'pdf'))
RULE, HEAD = '0.5pt solid #000000', '1pt solid #000000'


def run(cmd):
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if r.returncode:
        sys.exit('failed: %s\n%s%s' % (' '.join(cmd), r.stdout[-1500:], r.stderr[-1500:]))


def borders(path):
    with zipfile.ZipFile(path) as z:
        items = {n: z.read(n) for n in z.namelist()}
    c, hit = items['content.xml'].decode(), 0
    for name, props in (('TableRowCell', 'fo:border="%s" fo:padding="0.06cm 0.12cm"' % RULE),
                        ('TableHeaderRowCell',
                         'fo:border="%s" fo:border-bottom="%s" fo:background-color="#ececec" '
                         'fo:padding="0.06cm 0.12cm"' % (RULE, HEAD))):
        c, k = re.subn(r'(<style:style style:name="%s" style:family="table-cell">\s*'
                       r'<style:table-cell-properties )fo:border="none"( */>)' % name,
                       r'\g<1>%s\g<2>' % props, c, count=1)
        hit += k
    if hit != 2:
        sys.exit('cell styles matched %d/2; pandoc output structure may have changed' % hit)
    items['content.xml'] = c.encode()
    with zipfile.ZipFile(path + '.t', 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr(zipfile.ZipInfo('mimetype'), items.pop('mimetype'),
                   compress_type=zipfile.ZIP_STORED)
        for k, v in items.items():
            z.writestr(k, v)
    shutil.move(path + '.t', path)


for p in (ODT, DOCX, PDF):
    if os.path.exists(p):
        os.remove(p)
run(['pandoc', MD, '-o', ODT, '--reference-doc', REF])
borders(ODT)
run(['soffice', '--headless', '--convert-to', 'docx:MS Word 2007 XML',
     '--outdir', os.path.dirname(DOCX), ODT])
run(['soffice', '--headless', '--convert-to', 'pdf', '--outdir', os.path.dirname(PDF), DOCX])
info = subprocess.run(['pdfinfo', PDF], capture_output=True, text=True).stdout
print('\n'.join(l for l in info.splitlines() if l.startswith(('Pages', 'Page size'))))
print('wrote', DOCX, 'and', PDF)
