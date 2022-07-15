for file in *.gv
do
  dot -Tpng -Gdpi=150 "$file" -o "i$file.png"
done

