import os
import numpy as np
import pandas as pd

def generate_split_data(num_samples, split_name):
    # 명세서의 샘플 분포 비율 반영 (60%, 20%, 15%, 5%)
    counts = {
        0: int(num_samples * 0.60),
        1: int(num_samples * 0.20),
        2: int(num_samples * 0.15),
        3: num_samples - int(num_samples * 0.60) - int(num_samples * 0.20) - int(num_samples * 0.15)
    }
    
    # 명세서 가이드라인의 레이블별 피처 생성 범위 정의
    ranges = {
        0: {"rps": (0, 400),   "occ": (0, 40),   "loss": (0, 2),   "lat": (0, 50)},
        1: {"rps": (400, 650), "occ": (40, 65),  "loss": (2, 8),   "lat": (50, 150)},
        2: {"rps": (650, 850), "occ": (65, 85),  "loss": (8, 20),  "lat": (150, 300)},
        3: {"rps": (850, 1000),"occ": (85, 100), "loss": (20, 30), "lat": (300, 500)}
    }
    
    data_rows = []
    sample_id_offset = 0
    
    for label, count in counts.items():
        r = ranges[label]
        for _ in range(count):
            # 각 샘플당 10개의 타임스텝(시계열 윈도우) 생성
            for t in range(10):
                rps = np.random.uniform(*r["rps"])
                occ = np.random.uniform(*r["occ"])
                loss = np.random.uniform(*r["loss"])
                lat = np.random.uniform(*r["lat"])
                
                data_rows.append([sample_id_offset, t, rps, occ, loss, lat, label])
            sample_id_offset += 1
            
    df = pd.DataFrame(data_rows, columns=["sample_id", "timestep", "rps", "occupancy", "loss_rate", "latency", "label"])
    
    # 윈도우 내부 순서(timestep 0~9)를 유지하면서 샘플 묶음 자체를 무작위로 섞음
    unique_ids = df["sample_id"].unique()
    np.random.shuffle(unique_ids)
    
    shuffled_rows = []
    for new_id, old_id in enumerate(unique_ids):
        sample_df = df[df["sample_id"] == old_id].copy()
        sample_df["sample_id"] = new_id
        shuffled_rows.append(sample_df)
        
    final_df = pd.concat(shuffled_rows).sort_values(by=["sample_id", "timestep"]).reset_index(drop=True)
    
    # 코랩 환경 내에 폴더가 없다면 자동 생성
    os.makedirs("data/dummy", exist_ok=True)
    
    output_path = f"data/dummy/{split_name}.csv"
    final_df.to_csv(output_path, index=False)
    print(f"✓ 성공: {output_path} 생성 완료 | 총 샘플 수: {num_samples}")

if __name__ == "__main__":
    # 데이터 고정 생성을 위해 랜덤 시드 지정
    np.random.seed(42)
    
    generate_split_data(700, "train")  # 학습용 70%
    generate_split_data(150, "val")    # 검증용 15%
    generate_split_data(150, "test")   # 테스트용 15%
    print("더미 데이터 생성 완료")
