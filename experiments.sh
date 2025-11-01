data_list=(
    "FT_COCOGRAY" "FT_DiffV2IR"
)

# 出力ファイル名
output_file="results.csv"

# ファイルが存在しない場合のみヘッダを書き込む
if [ ! -f "$output_file" ]; then
    echo "Arch,Run,AP,AP.5,AP.75,AP(M),AP(L),AR,AR.5,AR.75,AR(M),AR(L)" > "$output_file"
fi

# データごとの処理
for data in "${data_list[@]}"; do
    tmpfile=$(mktemp)

    for i in $(seq 1 5); do
        echo "Running: $data - Run $i"
        python train.py --cfg experiments/"$data".yaml

        line=$(python eval.py --cfg experiments/"$data".yaml -m output/BaseData/pose_resnet_50/"$data"/best_epoch.pth | \
        grep -E 'Average Precision|Average Recall' | \
        grep -oE '[0-9.-]+$' | \
        paste -sd "," -)

        # 保存（形式: Arch名,Run番号,値...）
        echo "$data,Run$i,$line" >> "$tmpfile"
    done

    # 平均値の計算
    avg=$(awk -F, -v name="$data" '
    {
        for(i=3;i<=NF;i++) sum[i]+=$i
        count++
    }
    END {
        printf "%s,avg", name
        for(i=3;i<=NF;i++) {
            avg = sum[i]/count
            printf ",%.3f", avg
        }
        printf "\n"
    }' "$tmpfile")

    # 結果を追記
    cat "$tmpfile" >> "$output_file"
    echo "$avg" >> "$output_file"
    echo "" >> "$output_file"

    rm "$tmpfile"
done
