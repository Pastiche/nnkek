N_JOBS=50
OUTDIR=/data/shared/dataset/aer_user_images_ru/image
TXTDIR=/data/shared/dataset/aer_user_images_ru/txt

mkdir -p "$OUTDIR"

download_index() {
  filepath=$1/${3}.txt
  outdir=$2/${3}
  mkdir -p  "$outdir"
  echo $filepath
  echo $outdir
  aria2c --continue=true --dir=$outdir --input-file=$filepath
}

export -f download_index
parallel -j$N_JOBS --line-buffer download_index $TXTDIR $OUTDIR ::: {0..99}