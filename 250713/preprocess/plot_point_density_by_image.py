import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import Counter

def is_number(v):
    try:
        _ = float(v)
        return True
    except Exception:
        return False

def parse_building_floor_from_name(name):
    """
    依需求：
    - 前三個英文字母為建築 (不分大小寫)
    - 英文字母後第一個數字字元為樓層
    """
    if not isinstance(name, str):
        return None, None
    s = name.strip()
    if len(s) < 4:
        return None, None
    b = s[:3]
    if not b.isalpha():
        return None, None
    building = b.upper()
    # 直接取第4個字元，若非數字則往後尋找第一個數字
    floor = None
    if len(s) >= 4 and s[3].isdigit():
        floor = int(s[3])
    else:
        for ch in s[3:]:
            if ch.isdigit():
                floor = int(ch)
                break
    return building, floor

def collect_points_by_image(input_dir, recursive=True):
    """
    掃描 input_dir 內所有 .json 檔，依 imageId 聚合 (x, y)。
    回傳:
      points_by_image: dict[imageId] -> np.ndarray shape (N, 2)
      meta_by_image: dict[imageId] -> {
        'building': str|None, 'floor': int|None, 'building_counter': Counter,
        'wifi_total': int, 'wifi_valid': int,
        'wifi_by_building_valid': dict[str,int],
        'wifi_share_by_building': dict[str,float] (百分比)
      }
      file_count: 掃描到的 json 檔數
      wifi_totals: {'total': int, 'valid': int, 'points_with_wifi': int}
    """
    points_by_image = {}
    meta_counters = {}  # imageId -> {'b': Counter(), 'f': Counter()}
    wifi_totals = {'total': 0, 'valid': 0, 'points_with_wifi': 0}
    # 擴充：每張圖的 WiFi 計數，含建築別統計
    wifi_counts_by_image = {}  # imageId -> {'total': int, 'valid': int, 'by_building': {SEA|SEB|SEC: {'total','valid'}}}
    file_count = 0

    walker = os.walk(input_dir) if recursive else [(input_dir, [], os.listdir(input_dir))]
    for root, _, files in walker:
        for fname in files:
            if not fname.lower().endswith(".json"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as fin:
                    data = json.load(fin)
                file_count += 1
            except Exception as e:
                print(f"警告: 無法讀取 {fpath}: {e}")
                continue

            for item in data if isinstance(data, list) else []:
                img_id = item.get("imageId", None)
                x = item.get("x", None)
                y = item.get("y", None)
                # 先解析此點位的建築/樓層
                bld, flr = parse_building_floor_from_name(item.get("name", ""))

                # 統計此點的 WiFi RSSI 筆數
                readings = item.get("wifiReadings", [])
                if isinstance(readings, list):
                    total_here = len(readings)
                    valid_here = 0
                    for r in readings:
                        lv = r.get('level', None)
                        if is_number(lv):
                            valid_here += 1
                    wifi_totals['total'] += total_here
                    wifi_totals['valid'] += valid_here
                    if total_here > 0:
                        wifi_totals['points_with_wifi'] += 1
                    # 累積到 per-image（含建築別分流）
                    if img_id is not None:
                        key = str(img_id)
                        wc = wifi_counts_by_image.setdefault(key, {'total': 0, 'valid': 0, 'by_building': {}})
                        wc['total'] += total_here
                        wc['valid'] += valid_here
                        if bld:
                            bkey = bld.upper()
                            if bkey in ('SEA', 'SEB', 'SEC'):
                                bb = wc['by_building'].setdefault(bkey, {'total': 0, 'valid': 0})
                                bb['total'] += total_here
                                bb['valid'] += valid_here

                # 聚合座標與建築/樓層統計（需有效座標）
                if img_id is None or x is None or y is None or not (is_number(x) and is_number(y)):
                    continue
                img_id = str(img_id)
                if img_id not in points_by_image:
                    points_by_image[img_id] = []
                points_by_image[img_id].append([float(x), float(y)])

                ctr = meta_counters.setdefault(img_id, {'b': Counter(), 'f': Counter()})
                if bld:
                    ctr['b'][bld] += 1
                if flr is not None:
                    ctr['f'][int(flr)] += 1

    # 彙整 meta（含建築別 WiFi 統計與佔比）
    meta_by_image = {}
    for k in list(points_by_image.keys()):
        arr = np.asarray(points_by_image[k], dtype=float)
        if arr.size == 0:
            del points_by_image[k]
            meta_counters.pop(k, None)
            wifi_counts_by_image.pop(k, None)
        else:
            points_by_image[k] = arr
            b = meta_counters.get(k, {}).get('b', Counter())
            f = meta_counters.get(k, {}).get('f', Counter())
            building = b.most_common(1)[0][0] if b else None
            floor = f.most_common(1)[0][0] if f else None

            wc = wifi_counts_by_image.get(k, {'total': 0, 'valid': 0, 'by_building': {}})
            by_b = wc.get('by_building', {})
            # 取有效筆數為基礎計算佔比
            valid_sum = sum(v.get('valid', 0) for v in by_b.values()) or 0
            shares = {}
            for name in ('SEA', 'SEB', 'SEC'):
                vcnt = int(by_b.get(name, {}).get('valid', 0))
                shares[name] = (vcnt / valid_sum * 100.0) if valid_sum > 0 else 0.0

            meta_by_image[k] = {
                'building': building,
                'floor': floor,
                'building_counter': b,
                'wifi_total': int(wc.get('total', 0)),
                'wifi_valid': int(wc.get('valid', 0)),
                'wifi_by_building_valid': {
                    'SEA': int(by_b.get('SEA', {}).get('valid', 0)),
                    'SEB': int(by_b.get('SEB', {}).get('valid', 0)),
                    'SEC': int(by_b.get('SEC', {}).get('valid', 0)),
                },
                'wifi_share_by_building': shares  # 百分比
            }

    return points_by_image, meta_by_image, file_count, wifi_totals

def plot_density(xy, title, out_path, bins=100, scatter=False, dpi=150, cmap="hot"):
    """
    使用 2D 直方圖繪製密度圖。
    xy: np.ndarray (N, 2)
    """
    x = xy[:, 0]
    y = xy[:, 1]

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    # 避免全部點相同導致範圍為 0
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        y_max = y_min + 1.0

    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[x_min, x_max], [y_min, y_max]])

    # 繪圖
    fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(
        H.T,  # 注意轉置
        extent=extent,
        origin="lower",
        cmap=cmap,
        aspect="equal",
        interpolation="nearest"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Point density (counts per bin)")

    if scatter:
        ax.scatter(x, y, s=3, c="cyan", alpha=0.5, linewidths=0)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def save_summary(points_by_image, meta_by_image, output_dir, wifi_totals, summary_name="density_summary"):
    """
    輸出統計摘要 (JSON 與 CSV)，含建築與樓層資訊、各建築物點數分佈與 WiFi RSSI 筆數與份額
    """
    summary = {
        "total_images": len(points_by_image),
        "total_points": int(sum(v.shape[0] for v in points_by_image.values())),
        "total_wifi_rssi": int(wifi_totals.get('total', 0)),
        "valid_wifi_rssi": int(wifi_totals.get('valid', 0)),
        "points_with_wifi": int(wifi_totals.get('points_with_wifi', 0)),
        "images": []
    }
    for img_id, xy in points_by_image.items():
        x = xy[:, 0]
        y = xy[:, 1]
        meta = meta_by_image.get(img_id, {}) if meta_by_image else {}
        building_counter = meta.get('building_counter', Counter())
        building_breakdown = dict(building_counter) if building_counter else {}

        # 新增：各建築 WiFi 有效筆數與佔比
        by_valid = meta.get('wifi_by_building_valid', {})
        shares = meta.get('wifi_share_by_building', {})

        summary["images"].append({
            "imageId": img_id,
            "building": meta.get("building"),
            "floor": meta.get("floor"),
            "count": int(xy.shape[0]),
            "wifi_total": int(meta.get('wifi_total', 0)),
            "wifi_valid": int(meta.get('wifi_valid', 0)),
            "building_breakdown": building_breakdown,
            "building_wifi_valid_counts": {
                "SEA": int(by_valid.get("SEA", 0)),
                "SEB": int(by_valid.get("SEB", 0)),
                "SEC": int(by_valid.get("SEC", 0))
            },
            "building_wifi_shares_percent": {
                "SEA": float(shares.get("SEA", 0.0)),
                "SEB": float(shares.get("SEB", 0.0)),
                "SEC": float(shares.get("SEC", 0.0))
            },
            "x_min": float(np.min(x)),
            "x_max": float(np.max(x)),
            "y_min": float(np.min(y)),
            "y_max": float(np.max(y))
        })

    # JSON
    json_path = os.path.join(output_dir, f"{summary_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # CSV（新增各建築 WiFi 有效筆數與佔比欄位）
    csv_path = os.path.join(output_dir, f"{summary_name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "imageId", "building", "floor", "count",
            "wifi_total", "wifi_valid",
            "SEA_count", "SEB_count", "SEC_count",
            "SEA_wifi_valid", "SEB_wifi_valid", "SEC_wifi_valid",
            "SEA_wifi_share_pct", "SEB_wifi_share_pct", "SEC_wifi_share_pct",
            "x_min", "x_max", "y_min", "y_max"
        ])
        for it in summary["images"]:
            breakdown = it.get("building_breakdown", {})
            bw_valid = it.get("building_wifi_valid_counts", {})
            shares = it.get("building_wifi_shares_percent", {})
            writer.writerow([
                it["imageId"],
                it["building"],
                it["floor"],
                it["count"],
                it.get("wifi_total", 0),
                it.get("wifi_valid", 0),
                breakdown.get("SEA", 0),
                breakdown.get("SEB", 0),
                breakdown.get("SEC", 0),
                bw_valid.get("SEA", 0),
                bw_valid.get("SEB", 0),
                bw_valid.get("SEC", 0),
                f"{shares.get('SEA', 0.0):.2f}",
                f"{shares.get('SEB', 0.0):.2f}",
                f"{shares.get('SEC', 0.0):.2f}",
                it["x_min"], it["x_max"], it["y_min"], it["y_max"]
            ])

    print(f"已輸出統計摘要: {json_path}")
    print(f"已輸出統計摘要: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="依 imageId 繪製 scan13 點密度圖")
    parser.add_argument("--input_dir", type=str, default="../points/scan13", help="輸入 scan13 目錄")
    parser.add_argument("--output_dir", type=str, default="../plots/point_density", help="輸出圖檔目錄")
    parser.add_argument("--bins", type=int, default=100, help="2D 直方圖網格大小")
    parser.add_argument("--no-recursive", action="store_true", help="不遞迴搜尋子資料夾")
    parser.add_argument("--scatter", action="store_true", help="在密度圖上疊加散點")
    parser.add_argument("--dpi", type=int, default=150, help="輸出圖檔 DPI")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    recursive = not args.no_recursive
    points_by_image, meta_by_image, file_count, wifi_totals = collect_points_by_image(input_dir, recursive=recursive)
    if not points_by_image:
        print(f"在 {input_dir} 未找到可用點位，或缺少 imageId/x/y。")
        return

    print(f"已讀取 {file_count} 個 JSON 檔，偵測到 {len(points_by_image)} 個不同的 imageId。")
    print(f"總 WiFi RSSI 筆數: {wifi_totals.get('total', 0)}，有效數值: {wifi_totals.get('valid', 0)}，含有 WiFi 的點位數: {wifi_totals.get('points_with_wifi', 0)}")

    # 專門針對合併圖（SE，且樓層為 1/2/3）輸出 SEA/SEB/SEC 的 WiFi RSSI 佔比
    print("\n合併圖（SE）之各建築 WiFi RSSI 佔比：")
    for img_id, meta in meta_by_image.items():
        if str(meta.get('building', '')).upper() == 'SE' and str(meta.get('floor', '')) in {'1', '2', '3', 1, 2, 3}:
            shares = meta.get('wifi_share_by_building', {})
            by_valid = meta.get('wifi_by_building_valid', {})
            floor = meta.get('floor')
            print(f"- imageId={img_id} F{floor}: "
                  f"SEA {shares.get('SEA', 0.0):.2f}% ({by_valid.get('SEA', 0)}), "
                  f"SEB {shares.get('SEB', 0.0):.2f}% ({by_valid.get('SEB', 0)}), "
                  f"SEC {shares.get('SEC', 0.0):.2f}% ({by_valid.get('SEC', 0)})")

    for img_id, xy in points_by_image.items():
        meta = meta_by_image.get(img_id, {})
        tag = f"{meta.get('building','?')}-F{meta.get('floor','?')}"
        out_png = os.path.join(output_dir, f"density_image_{img_id}.png")
        title = f"Point Density - imageId={img_id} [{tag}] (N={xy.shape[0]})"
        plot_density(xy, title, out_png, bins=args.bins, scatter=args.scatter, dpi=args.dpi)
        print(f"已輸出圖像: {out_png}")

    save_summary(points_by_image, meta_by_image, output_dir, wifi_totals)

if __name__ == "__main__":
    main()
