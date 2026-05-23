import json
from collections import Counter

from src.safe_io import read_limited_json

data = read_limited_json('validation/geokerr_strict_comparison.json')

print('Status confusion matrix:')
print('  geokerr -> cpu')
counts = Counter((r['geokerr_status'], r['cpu_status']) for r in data['per_ray'])
for (g, c), n in sorted(counts.items()):
    print(f'  {g:10s} -> {c:10s}: {n:3d}')

print()
print('Sample mismatches (captured->disk):')
shown = 0
for r in data['per_ray']:
    if r['geokerr_status'] == 'captured' and r['cpu_status'] == 'disk':
        print(f'  alpha={r["alpha"]:6.1f} beta={r["beta"]:6.1f}  geo_min_r={r["geokerr_min_r"]:10.3f} cpu_r={r["cpu_final_r"]:10.3f}')
        shown += 1
        if shown >= 5:
            break

print()
print('Sample mismatches (disk->captured):')
shown = 0
for r in data['per_ray']:
    if r['geokerr_status'] == 'disk' and r['cpu_status'] == 'captured':
        print(f'  alpha={r["alpha"]:6.1f} beta={r["beta"]:6.1f}  geo_min_r={r["geokerr_min_r"]:10.3f} cpu_r={r["cpu_final_r"]:10.3f}')
        shown += 1
        if shown >= 5:
            break
