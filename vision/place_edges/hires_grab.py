"""One HD1200 snapshot from the nano gstsender HIRES_PORT -> hires.png + depth_hi.npy (uint16 mm)."""
import socket, struct, sys
import cv2, numpy as np


def read_hires(f):
    w, h, npng, ndep = struct.unpack(">HHII", f.read(12))
    png, dep = f.read(npng), f.read(ndep)
    if len(png) < npng or len(dep) < ndep:
        raise IOError("short hi-res frame")
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR), np.frombuffer(dep, "<u2").reshape(h, w)


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:  # protocol round trip, no camera
        import io
        L = np.random.default_rng(0).integers(0, 255, (12, 16, 3), dtype=np.uint8); D = np.arange(192, dtype="<u2").reshape(12, 16)
        png = cv2.imencode(".png", L)[1]
        im, d = read_hires(io.BytesIO(struct.pack(">HHII", 16, 12, png.nbytes, D.nbytes) + png.tobytes() + D.tobytes()))
        assert (im == L).all() and (d == D).all(); print("selftest ok"); sys.exit()
    with socket.create_connection(("192.168.50.22", 55704), timeout=10) as s:
        im, d = read_hires(s.makefile("rb"))
    cv2.imwrite("hires.png", im); np.save("depth_hi.npy", d)
    v = d[d > 0]; print("hires", im.shape, "depth valid %.0f%%" % (100 * v.size / d.size))
