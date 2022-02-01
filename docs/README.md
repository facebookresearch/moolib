# moolib Documentation

### Run Docsite Locally
```
pip install sphinx==4.1.2
./run_docs.sh
```

### To Update Jeckyll Site
```
make html
mv build/html/* .
rm -r build/doctrees/
touch .nojekyll
```
And commit and push to branch `gh-pages`
