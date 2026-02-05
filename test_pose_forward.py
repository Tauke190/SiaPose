#!/usr/bin/env python
"""
Test script to verify the pose detection forward pass works correctly.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

def test_pose_forward():
    """Test the forward pass of SIA with pose detection enabled."""
    from sia import SIA

    print("=" * 60)
    print("Testing SIA with Pose Detection")
    print("=" * 60)

    # Test parameters
    batch_size = 2
    num_frames = 9
    height = 224
    width = 224
    det_token_num = 100
    num_keypoints = 17

    # Create model (without pretrained weights for testing)
    print("\n1. Creating SIA model with pose detection...")
    model = SIA(
        size='b',  # Use smaller model for testing
        pretrain=None,  # No pretrained weights
        det_token_num=det_token_num,
        num_frames=num_frames,
        num_keypoints=num_keypoints,
        pose_decoder_layers=2,
        enable_pose=True,
    )
    model.eval()
    print(f"   Model created successfully!")
    print(f"   - Vision encoder: {model.vision_encoder_name}")
    print(f"   - Det tokens: {model.det_token_num}")
    print(f"   - Keypoints: {model.num_keypoints}")
    print(f"   - Pose enabled: {model.enable_pose}")

    # Create dummy input
    print("\n2. Creating dummy input...")
    x = torch.randn(batch_size, num_frames, 3, height, width)
    print(f"   Input shape: {x.shape}")

    # Forward pass
    print("\n3. Running forward pass...")
    with torch.no_grad():
        output = model(x)

    # Check outputs
    print("\n4. Checking output shapes...")
    print(f"   Output keys: {list(output.keys())}")

    assert 'pred_logits' in output, "Missing pred_logits"
    assert 'pred_boxes' in output, "Missing pred_boxes"
    assert 'human_logits' in output, "Missing human_logits"
    assert 'pred_keypoints' in output, "Missing pred_keypoints"

    print(f"   pred_logits shape: {output['pred_logits'].shape}")
    print(f"   pred_boxes shape: {output['pred_boxes'].shape}")
    print(f"   human_logits shape: {output['human_logits'].shape}")
    print(f"   pred_keypoints shape: {output['pred_keypoints'].shape}")

    # Verify shapes
    assert output['pred_logits'].shape[:2] == (batch_size, det_token_num), \
        f"Unexpected pred_logits shape: {output['pred_logits'].shape}"
    assert output['pred_boxes'].shape == (batch_size, det_token_num, 4), \
        f"Unexpected pred_boxes shape: {output['pred_boxes'].shape}"
    assert output['human_logits'].shape == (batch_size, det_token_num, 2), \
        f"Unexpected human_logits shape: {output['human_logits'].shape}"
    assert output['pred_keypoints'].shape == (batch_size, det_token_num, num_keypoints, 3), \
        f"Unexpected pred_keypoints shape: {output['pred_keypoints'].shape}"

    print("\n5. Verifying output ranges...")
    print(f"   pred_boxes range: [{output['pred_boxes'].min():.4f}, {output['pred_boxes'].max():.4f}]")
    print(f"   pred_keypoints xy range: [{output['pred_keypoints'][..., :2].min():.4f}, {output['pred_keypoints'][..., :2].max():.4f}]")
    print(f"   pred_keypoints vis range: [{output['pred_keypoints'][..., 2].min():.4f}, {output['pred_keypoints'][..., 2].max():.4f}]")

    # Boxes and keypoints should be in [0, 1] range (sigmoid applied)
    assert output['pred_boxes'].min() >= 0 and output['pred_boxes'].max() <= 1, \
        "pred_boxes should be in [0, 1] range"
    assert output['pred_keypoints'][..., :2].min() >= 0 and output['pred_keypoints'][..., :2].max() <= 1, \
        "pred_keypoints xy should be in [0, 1] range"
    assert output['pred_keypoints'][..., 2].min() >= 0 and output['pred_keypoints'][..., 2].max() <= 1, \
        "pred_keypoints visibility should be in [0, 1] range"

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

    return True


def test_pose_disabled():
    """Test that model works with pose detection disabled."""
    from sia import SIA

    print("\n" + "=" * 60)
    print("Testing SIA with Pose Detection DISABLED")
    print("=" * 60)

    model = SIA(
        size='b',
        pretrain=None,
        det_token_num=100,
        num_frames=9,
        enable_pose=False,  # Disable pose
    )
    model.eval()

    x = torch.randn(2, 9, 3, 224, 224)

    with torch.no_grad():
        output = model(x)

    print(f"   Output keys: {list(output.keys())}")

    # Should NOT have pred_keypoints when pose is disabled
    assert 'pred_keypoints' not in output, "pred_keypoints should not be present when pose is disabled"

    print("   Pose disabled test passed!")
    return True


if __name__ == "__main__":
    test_pose_forward()
    # test_pose_disabled()