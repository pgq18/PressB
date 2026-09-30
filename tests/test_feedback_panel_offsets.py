"""Button feedback ROI follows XY translations; legacy callers stay centered."""
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip('av')
pytest.importorskip('cv2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_button_feedback as audit


def test_feedback_projection_preserves_legacy_and_applies_both_axes(monkeypatch):
    seen=[]
    def projection(points,camera,intrinsics):
        seen.append(points.copy())
        return np.array([[10,10],[10,20],[20,10],[20,20]]),np.ones(4)
    monkeypatch.setattr(audit.original,'project',projection)
    cfg=dict(button_face_x=.46,button_column_y=.045,button_bottom_z=.98,button_pitch_z=.035)
    args=(np.zeros((480,640,3),dtype=np.uint8),0,0.,0,.001,33,cfg,np.eye(4),np.eye(3))
    legacy=audit.light_sample(*args)
    explicit=audit.light_sample(*args,panel_offset_x_m=0.,panel_offset_y_m=0.)
    shifted=audit.light_sample(*args,panel_offset_x_m=.008,panel_offset_y_m=-.025)
    assert legacy==explicit==shifted
    np.testing.assert_array_equal(seen[0],seen[1])
    np.testing.assert_allclose(seen[2]-seen[0],np.broadcast_to([.008,-.025,0.],(4,3)),atol=1e-15)
