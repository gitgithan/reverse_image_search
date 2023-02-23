mkdir -p datasets

if [ "$1" == "caltech" ] || [ "$1" == "all" ]; then
    # caltech 101 dataset: https://en.wikipedia.org/wiki/Caltech_101
    echo 'Start downloading caltech101'
    gdown "https://drive.google.com/uc?id=137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp&confirm=t" --output datasets/caltech101.tar.gz
    tar -xzf datasets/caltech101.tar.gz --directory datasets

    mv datasets/101_ObjectCategories datasets/caltech101 
    rm -rf datasets/caltech101/101_ObjectCategories
    rm datasets/caltech101.tar.gz
    echo 'Finished downloading caltech101'
fi

if [ "$1" == "voc" ] || [ "$1" == "all" ]; then # VOC2012 takes ~ 6 min TO DOWNLOAD 
    # voc2012 dataset: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
    echo 'Start downloading voc2012'
    wget -P datasets "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    tar -xf datasets/VOCtrainval_11-May-2012.tar --directory datasets
    rm datasets/VOCtrainval_11-May-2012.tar
    echo 'Finished downloading voc2012'
fi