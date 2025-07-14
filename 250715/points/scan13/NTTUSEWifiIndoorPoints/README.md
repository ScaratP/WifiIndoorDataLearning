# 專案概述

本專案旨在建立一個基於 WiFi 訊號的室內定位和導航系統，用於校園環境中引導使用者找到教室和其他地點。專案利用採集的 WiFi 數據訓練定位模型，並使用 Android 應用程式提供導航功能。

### 檔案說明
專案中的關鍵檔案包括：

- classroom_points.json：包含教室位置點的數據，用於 Android 應用程式中建立導航路線
- se_all_13.json：包含所有蒐集的點位資料和 WiFi 訊號強度數據
- se_all_13_filtered.json：經過處理的 WiFi 數據，只留下名稱為Ap-NTTU、Ap2-NTTU、NTTU、eduroam
- filter_wifi.py：用於過濾原始 WiFi 訊號數據的腳本
- filter_by_name_length.py：根據名稱特性進行資料過濾的腳本

### 實作指南
導入 classroom_points.json 到 Android Studio

1. 在 Android 專案中，創建 assets 資料夾（如果尚未存在）
2. 將 classroom_points.json 複製到 assets 資料夾中
3. 在應用程式中使用以下代碼讀取點位資料：
```kotlin
try {
    InputStream is = getAssets().open("classroom_points.json");
    int size = is.available();
    byte[] buffer = new byte[size];
    is.read(buffer);
    is.close();
    String json = new String(buffer, "UTF-8");
    // 使用 JSON 解析庫（如 Gson）解析數據
    Gson gson = new Gson();
    PointData[] points = gson.fromJson(json, PointData[].class);
    // 可以開始使用點位數據
} catch (IOException e) {
    e.printStackTrace();
}
```
4. 使用內建導航元件（如 Google Maps Indoor 或自定義地圖視圖）建立路線
5. 在地圖上顯示教室點位，並實現路徑規劃演算法

### 訓練室內定位模型
1. 使用 se_all_13.json 和 se_all_13_filtered.json 作為訓練數據集
2. 實現指紋定位或三邊測量等室內定位演算法
3. 使用機器學習模型（如 kNN、隨機森林或神經網絡）提高定位準確度
4. 將訓練好的模型整合到 Android 應用程式中


# 未來計劃
### 模型優化
- 使用更多採集點來提高模型準確度
- 實驗不同的機器學習演算法，比較定位效果
- 結合加速度計、陀螺儀等傳感器數據改善定位精度

### 使用者介面改進
- 設計更直觀的室內地圖展示
- 增加搜尋教室和其他設施的功能
- 實現不同樓層間的無縫導航
- 加入位置分享功能，方便使用者群組相互定位

### 待辦事項
<input disabled="" type="checkbox"> 編寫 Android 應用程式讀取 classroom_points.json 的功能

<input disabled="" type="checkbox"> 實現基本的室內地圖顯示功能

<input disabled="" type="checkbox"> 開發使用 se_all_13_filtered.json 進行定位的演算法

<input disabled="" type="checkbox"> 建立簡單的路徑規劃功能，連接兩點之間的最短路徑

<input disabled="" type="checkbox"> 測試不同定位演算法在室內環境的表現

<input disabled="" type="checkbox"> 開發使用者交互功能，如點選地點、搜尋功能等

<input disabled="" type="checkbox"> 優化 WiFi 指紋數據庫，提高定位精確度

<input disabled="" type="checkbox"> 構建應用程式部署流程

<input disabled="" type="checkbox"> 進行現場測試，評估系統性能

<input disabled="" type="checkbox"> 根據測試結果修改和優化系統