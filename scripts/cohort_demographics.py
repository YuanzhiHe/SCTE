"""Aggregate demographics for the cohort description sentence.

The prepared pairs carry no patient identity: PROTOCOL_TAGS in prep_pairs.py is an
allow-list of acquisition parameters, and the 0010 group is excluded on purpose, so
age and sex have to come from the source DICOM.

This reads those headers on the hospital machine and prints ONLY aggregate numbers.
Nothing per-patient is written to disk or to the output, so the printed block is the
only thing that needs to leave, and it carries no identifier.

It also counts DISTINCT PatientIDs, which the manuscript needs separately from the
examination count: the paired tests treat each examination as independent, and that
holds only if no patient contributed two.

  python scripts/cohort_demographics.py --dicom /data/henan --cases PRIVATE/hebei/pairs_external/test
"""
import argparse, os, sys, collections
import numpy as np

try:
    import pydicom
except ImportError:
    sys.exit('pydicom is required: pip install pydicom')

ap = argparse.ArgumentParser()
ap.add_argument('--dicom', required=True, help='root of the source DICOM tree')
ap.add_argument('--cases', default=None,
                help='prepared-pair directory; restricts the count to the reported '
                     'cases rather than everything under --dicom')
ap.add_argument('--id_map', default=None,
                help='ID_MAP_DO_NOT_EXPORT.csv, if the prepared case ids are hashes. '
                     'Read locally and never printed.')
a = ap.parse_args()

wanted = None
if a.cases:
    wanted = {f[:-9] for f in os.listdir(a.cases) if f.endswith('_thin.npy')}
    if a.id_map and os.path.exists(a.id_map):
        import csv
        m = {r['case']: r.get('patient_id') or r.get('study_uid')
             for r in csv.DictReader(open(a.id_map))}
        wanted = {m[c] for c in wanted if c in m}
    print(f'restricting to {len(wanted)} reported cases')


def age_years(ds):
    """PatientAge is usually nnnY but may be blank; fall back to the dates."""
    v = getattr(ds, 'PatientAge', '') or ''
    v = v.strip()
    if v.endswith('Y') and v[:-1].isdigit():
        return int(v[:-1])
    if v.isdigit():
        return int(v)
    b = getattr(ds, 'PatientBirthDate', '') or ''
    d = getattr(ds, 'StudyDate', '') or getattr(ds, 'AcquisitionDate', '') or ''
    if len(b) == 8 and len(d) == 8:
        y = int(d[:4]) - int(b[:4]) - ((int(d[4:8]) < int(b[4:8])) and 1 or 0)
        if 0 < y < 120:
            return y
    return None


seen, ages, sexes = {}, [], collections.Counter()
n_files = 0
for root, _, files in os.walk(a.dicom):
    for f in files:
        p = os.path.join(root, f)
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
        except Exception:
            continue
        pid = getattr(ds, 'PatientID', None)
        study = getattr(ds, 'StudyInstanceUID', None)
        if not pid or not study:
            continue
        n_files += 1
        if wanted is not None and pid not in wanted and study not in wanted:
            continue
        if study in seen:
            continue
        seen[study] = pid
        y = age_years(ds)
        if y is not None:
            ages.append(y)
        s = (getattr(ds, 'PatientSex', '') or '').strip().upper()
        sexes[s if s in ('M', 'F') else 'unknown'] += 1

n_exam = len(seen)
n_pat = len(set(seen.values()))
print(f'\nDICOM files read            {n_files}')
print(f'examinations (distinct study UID)  {n_exam}')
print(f'patients (distinct PatientID)      {n_pat}')
if n_exam != n_pat:
    print(f'  -> {n_exam - n_pat} examinations share a patient with another; the paired\n'
          f'     tests assume independence, so this has to be stated or handled')
if ages:
    ages = np.array(ages)
    print(f'age, years                  mean {ages.mean():.0f} +- {ages.std(ddof=1):.0f} (SD), '
          f'range {ages.min()}-{ages.max()}, n = {len(ages)}')
else:
    print('age                         not recoverable from these headers')
print(f'sex                         {sexes.get("M",0)} men, {sexes.get("F",0)} women'
      + (f', {sexes["unknown"]} unrecorded' if sexes.get('unknown') else ''))
print(f'\nSentence: "{n_exam} examinations in {n_pat} patients '
      + (f'(mean age, {ages.mean():.0f} years +- {ages.std(ddof=1):.0f} [SD]; '
         f'{sexes.get("M",0)} men, {sexes.get("F",0)} women)"' if len(ages) else '(age unavailable)"'))
