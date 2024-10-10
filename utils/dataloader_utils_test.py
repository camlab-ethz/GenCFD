import unittest
import torch

from utils.dataloader_utils import (
    downsample, 
    upsample, 
    translate_horizontally_periodic_batched, 
    translate_horizontally_periodic_unbatched,
    StatsRecorder
)

class UtilFunctions(unittest.TestCase):

    def test_downsample(self):
        N_old = 16
        N_new = 8
        bs = 2
        c = 3

        u_ = torch.randn((bs, c, N_old, N_old))
        u_down = downsample(u_, N_new)

        expected_shape = (bs, c, N_new, N_new)
        self.assertEqual(
            u_down.shape, expected_shape, 
            f"Expected shape {expected_shape} but got {u_down.shape}"
        )

        self.assertFalse(torch.isnan(u_down).any(), "Output contains NaN values")
        self.assertFalse(torch.isinf(u_down).any(), "Output contains Inf values")

    def test_upsample(self):
        # Test parameters
        N_old = 8  # Original size
        N_new = 16  # Desired upsample size
        batch_size = 2
        channels = 3
        
        u = torch.randn((batch_size, channels, N_old, N_old), dtype=torch.float32)
    
        u_up = upsample(u, N_new)
        
        expected_shape = (batch_size, channels, N_new, N_new)
        self.assertEqual(u_up.shape, expected_shape, 
                         f"Expected shape {expected_shape} but got {u_up.shape}")
        
        self.assertFalse(torch.is_complex(u_up), "Expected real-valued tensor after upsampling")
        
        self.assertFalse(torch.isnan(u_up).any(), "Output contains NaN values")
        self.assertFalse(torch.isinf(u_up).any(), "Output contains Inf values")
    
    def test_translate_horizontally_periodic(self):

        N_grid = 8  # Original size
        bs = 2
        c = 3
        
        tensor = torch.arange(bs*c*N_grid*N_grid*2, dtype=torch.float32).reshape(bs, c, N_grid, N_grid*2)

        pixels = 1
        tensor_unbatched = translate_horizontally_periodic_unbatched(tensor[0,...], pixels, -1)
        tensor_batched = translate_horizontally_periodic_batched(tensor, pixels, -1)

        expected_batched = torch.roll(tensor, shifts=pixels, dims=-1)
        expected_unbatched = torch.roll(tensor[0,...], shifts=pixels, dims=-1)

        self.assertTrue(torch.equal(tensor_batched, expected_batched), "Batched translation failed")
        self.assertTrue(torch.equal(tensor_unbatched, expected_unbatched), "Unbatched translation failed")


class TestStatsRecorder(unittest.TestCase):

    def test_update(self):

        spatial_ax = (1, 2)
        compute_also_high_mom = False  # Can set to True to test higher moments
        idx_wassertain_ = torch.tensor([0, 1])
        num_batches = 5
        batch_to_keep = 1
        
        # Initialize the StatsRecorderNew instance
        stats_recorder = StatsRecorder(
            idx_wassertain_=idx_wassertain_, 
            num_batches=num_batches, 
            compute_also_high_mom=compute_also_high_mom, 
            batch_to_keep=batch_to_keep, 
            spatial_ax=spatial_ax
        )
        
        data_shape = (10, 3, 16, 16) 
        data = torch.randn(data_shape)  

        stats_recorder.update(data)
        
        # Assert basic properties
        assert stats_recorder.nobservations == 10, "Number of observations should be updated."
        assert stats_recorder.ndimensions == 16, "Number of dimensions should be 3." # should be 3
        
        # Test if mean and std are computed correctly
        assert torch.allclose(stats_recorder.mean, data.mean(dim=0)), "Mean should be calculated correctly."
        assert torch.allclose(stats_recorder.std, data.std(dim=0)), "Standard deviation should be calculated correctly."
        
        # Test if min and max are computed correctly
        assert torch.all(stats_recorder.min == data.amin(dim=(0, 1, 2))), "Min value should be calculated correctly."
        assert torch.all(stats_recorder.max == data.amax(dim=(0, 1, 2))), "Max value should be calculated correctly."
        

        print("All assertions passed for the first update.")

        # Test updating with additional data
        new_data = torch.randn(data_shape)
        stats_recorder.update(new_data)

        # Check if the number of observations is accumulated
        assert stats_recorder.nobservations == 20, "Number of observations should accumulate after second update."


if __name__=="__main__":
    unittest.main()