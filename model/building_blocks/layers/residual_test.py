# Copyright 2024 The swirl_dynamics Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import residual
import torch
import unittest


class ResidualLayersTest(unittest.TestCase):

  def test_combine_res_with_skip_output_shape(self):
    test_cases = [
      ((8, 8, 8, 8), True), 
      ((8, 8, 8, 8, 8), False)
    ]
    for input_shape, project_skip in test_cases:
      with self.subTest(input_shape=input_shape, project_skip=project_skip):

        skip = res = torch.ones(input_shape)
        model = residual.CombineResidualWithSkip(project_skip=project_skip)
        out = model(res, skip)
        
        self.assertEqual(out.shape, input_shape)
        # If project_skip = False, variables should be an empty dict.
        if not project_skip:
          self.assertFalse(model.skip_projection)
        else:
          self.assertTrue(model.skip_projection)


if __name__ == "__main__":
  unittest.main()
