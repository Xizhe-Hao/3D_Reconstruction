"""Check serialized constraints across every exported contact surface."""
import json,re,os
from pathlib import Path
import numpy as np
from surface_roi_experiment import Region
from generate_contact_surface import tests
root=Path('outputs/rgbd_20261014')/os.environ.get('CONTACT_OUTPUT','surface_contact');m=json.loads((root/'manifest.json').read_text());cal=m['calibration'];reg=Region(json.loads(Path(os.environ.get('SURFACE_GRID','data/drive_test/capture_20261014_202456/sensor_grid.json')).read_text()));tests();assert [f['frame'] for f in m['frames']]==list(range(871));active=0;reclassified=0;max_plane=0.;max_travel=0.
for f in m['frames']:
 stem=f"frame_{f['frame']:05d}";d=np.load(root/(stem+'.npz'));o=d['observed'];i=d['inferred'];c=d['contact'];t=d['transition'];h=d['completed_height_mm'];assert not(o&i).any();assert np.array_equal(i,c|t);assert np.isfinite(h[o|i]).all();assert np.array_equal(h[o],d['height_mm'][o]);assert d['domain'][o|i].all();reclassified+=int(d['reclassified_observed'].sum())
 if f['contact_active']:
  active+=1;max_plane=max(max_plane,float(np.max(abs(h[c]-f['indenter_bottom_mm']))));pred=cal['params'][-1]+(f['force_distance_mm']-.05)*reg.axes[2,2];max_travel=max(max_travel,abs(pred-f['indenter_bottom_mm']))
 else:assert not c.any()
 j=json.loads((root/(stem+'.json')).read_text());assert np.asarray(j['projected_bottom_polygons']).shape==(4,4,2);assert j['frame']==f['frame'];assert np.isfinite(j['projected_bottom_polygons']).all()
 for kind,expected in [('observed',o.sum()),('transition',t.sum()),('contact',c.sum()),('display',(o|i).sum())]:
  b=(root/(stem+'_'+kind+'.ply')).read_bytes();end=b.index(b'end_header\n')+11;n=int(re.search(rb'element vertex (\d+)',b[:end])[1]);assert n==expected and len(b)-end==15*n
assert max_plane<1e-9 and max_travel<1e-9
r=dict(passed=True,frames=871,ply_files=3484,contact_frames=active,max_contact_plane_error_mm=max_plane,max_travel_equation_error_mm=max_travel,reclassified_observations_total=reclassified,checks=['flat rigid plane','travel consistency','no contact without travel','observations preserved outside inferred band','disjoint masks','ROI','PLY serialization','video projection metadata'],limitations='Numerical consistency checks; not independent physical accuracy.')
(root/'validation.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
