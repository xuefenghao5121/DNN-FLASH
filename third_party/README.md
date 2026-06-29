# third_party

This directory holds optional local copies of oneDNN for development.

## oneDNN dependency

FlashOne requires [oneDNN](https://github.com/oneapi-src/oneDNN) >= 3.0 for the
oneDNN integration layer. The library is **not** bundled in this repository.

### Install oneDNN

**Ubuntu/Debian:**

```bash
sudo apt install libdnnl-dev
```

**Build from source (local prefix):**

```bash
git clone https://github.com/oneapi-src/oneDNN.git
cd oneDNN && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../third_party/onednn-local/usr
make -j && make install
```

### CMake lookup order

`cmake/FindLocalDnnl.cmake` searches in this order:

1. `third_party/onednn-debian-3.12/usr` (if present, gitignored)
2. `third_party/onednn-local/usr` (if present, gitignored)
3. `/usr` and `/usr/local` (system install)
4. Default system search paths

If oneDNN is not found, the build skips oneDNN targets and falls back to the
reference path.
