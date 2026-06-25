# Third-party local dependencies

## oneDNN local package extraction

The current development machine does not allow sudo elevation from this Feishu session. Instead of installing system packages, FlashOne uses a local, project-scoped oneDNN extraction:

```bash
mkdir -p third_party/debs third_party/onednn-local
cd third_party/debs
apt-get download libdnnl-dev libdnnl3
for deb in *.deb; do dpkg-deb -x "$deb" ../onednn-local; done
```

This provides:

- `third_party/onednn-local/usr/include/dnnl.h`
- `third_party/onednn-local/usr/include/dnnl.hpp`
- `third_party/onednn-local/usr/include/oneapi/dnnl/dnnl.hpp`
- `third_party/onednn-local/usr/lib/x86_64-linux-gnu/libdnnl.so*`

`cmake/FindLocalDnnl.cmake` can find either this local extraction or a system oneDNN installation.
