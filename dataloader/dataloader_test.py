from torch.utils.data import DataLoader
import os
from dataloader.dataloader import DummyDataloader
from utils.model_utils import reshape_jax_torch

def test_cloudshock_dataset():
    cwd = os.getcwd()
    cwd = os.path.join(cwd, 'dataset')
    file = "cloud_shock_all_128.nc"
    file_path = os.path.join(cwd, file)


    small_dataset = (0, 100)  # Start and end indices for a partial dataset

    try:
        
        cloud_shock_dataset = DummyDataloader(file_path=file_path, partial=small_dataset)


        data_item = cloud_shock_dataset.__getitem__(idx=0)
        print(f"Shape of data item at index 0: {data_item.shape}")

        cloud_shock_data = reshape_jax_torch(cloud_shock_dataset.data_tensor)
        print(f"Reshaped data tensor shape: {cloud_shock_data.shape}")

        batch_size = 32

        train_dataloader = DataLoader(
            dataset=cloud_shock_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0 
        )

        # Iterate through one batch of the dataloader
        for i, batch in enumerate(train_dataloader):
            print(f"Batch {i} with shape {batch.shape}")
            # Exit after processing one batch for testing purposes
            if i == 0:
                break

        print("Test completed successfully!")

    except Exception as e:
        print(f"An error occurred during testing: {e}")

if __name__ == "__main__":
    test_cloudshock_dataset()

