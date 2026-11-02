"""Store large weights as float16 and cast to float32 at run time (browser-friendly size, same maths)."""
import sys, onnx, numpy as np
from onnx import numpy_helper, TensorProto, helper
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
new_inits, new_nodes, rename = [], [], {}
for t in g.initializer:
    a = numpy_helper.to_array(t)
    if a.dtype == np.float32 and a.size >= 4096:
        h = numpy_helper.from_array(a.astype(np.float16), t.name + "__fp16")
        new_inits.append(h)
        new_nodes.append(helper.make_node("Cast", [h.name], [t.name], to=TensorProto.FLOAT, name="cast_" + t.name))
    else:
        new_inits.append(t)
del g.initializer[:]
g.initializer.extend(new_inits)
nodes = list(g.node)
del g.node[:]
g.node.extend(new_nodes + nodes)     # casts first, then the original graph
onnx.checker.check_model(m)
onnx.save(m, dst)
import os; print(f"{os.path.getsize(dst)/1e6:.0f} MB")
