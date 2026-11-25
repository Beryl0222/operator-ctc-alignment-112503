"""验证业务输入之间的引用关系及形状，不计算目标输出。"""
import numpy as np
from support.inspect_data import load_bundle

def test_linked_materials():
    _, arrays, r, scenarios = load_bundle()
    shape = arrays['logits'].shape
    assert len(r['recordings']) == shape[0]
    for recording in r['recordings']:
        assert 0<=recording['valid_frames']<=shape[1]
        assert all(0<t<shape[2] for t in recording['targets'])
    assert all(all(str(t) in r['vocabulary']['tokens'] for t in p) for p in r['allowed_prefixes'])
