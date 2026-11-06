"""Lossless, explicitly negotiated camera transport and preprocessing."""
import base64
from copy import deepcopy
from io import BytesIO

import numpy as np
from PIL import Image
import pytest

from pressb.runtime.contract import decode_image, encode_image
from pressb.runtime.image_transport import (
    encode_preprocessed_image, prepare_wire_payload, resize_for_transport,
    validate_image_preprocessing, validate_preprocessing_marker, walloss_resize_spec,
)


def metadata():
    return {"image_encodings": ["png_base64", "png_bytes"], "observation": {"cameras": ["up"]},
            "image_preprocessing": walloss_resize_spec(256)}


def payload(size=(640, 480)):
    image = np.random.default_rng(42).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    return {"observations": [{"state": [0.], "task": "move", "images": {"up": encode_image(image)}}],
            "seeds": [5]}


def test_binary_png_removes_only_base64_without_decode(monkeypatch):
    value = payload()
    original = deepcopy(value)
    def unexpected(*args, **kwargs):
        pytest.fail("Binary-only negotiation must not decode or re-encode PNG")
    monkeypatch.setattr("pressb.runtime.image_transport.decode_image", unexpected)
    result = prepare_wire_payload("/encode", value, metadata(), image_transport="auto")
    image = result["observations"][0]["images"]["up"]
    assert image["encoding"] == "png_bytes"
    assert image["data"] == base64.b64decode(original["observations"][0]["images"]["up"]["data"])
    assert value == original


@pytest.mark.parametrize("size", [(640, 480), (480, 640), (319, 173), (71, 71), (32, 24)])
def test_pre_resize_exactly_matches_wall_pil_defaults_and_second_resize_identity(size):
    value = payload(size)
    original = deepcopy(value)
    image = decode_image(value["observations"][0]["images"]["up"])
    # These are the literal first-stage formulas in Wall's process_images.
    width, height = size
    target = (256, int(256 * height / width)) if width > height else (int(256 * width / height), 256)
    expected = image.resize(target)
    result = prepare_wire_payload("/infer", value, metadata(), image_transport="auto", preprocess_images=True)
    wire_image = result["observations"][0]["images"]["up"]
    received = decode_image(wire_image, require_rgb=True)
    np.testing.assert_array_equal(received, expected)
    np.testing.assert_array_equal(resize_for_transport(received, walloss_resize_spec(256)), expected)
    assert wire_image["preprocessing"] == {"spec": walloss_resize_spec(256), "original_size": list(size)}
    validate_preprocessing_marker(wire_image, received, metadata()["image_preprocessing"])
    assert value == original
    # Repeated forwarding of an already marked image is byte-exact.
    repeated = prepare_wire_payload("/infer", result, metadata(), image_transport="auto", preprocess_images=True)
    assert repeated == result


def test_negotiation_keeps_legacy_clients_and_unknown_backends_unchanged():
    value = payload()
    assert prepare_wire_payload("/encode", value, metadata()) is value
    assert prepare_wire_payload("/encode", value, {}, image_transport="auto", preprocess_images=True) is value
    assert prepare_wire_payload("/decode", value, metadata(), image_transport="auto", preprocess_images=True) is value
    result = prepare_wire_payload("/encode", value, metadata(), preprocess_images=True)
    assert result["observations"][0]["images"]["up"]["encoding"] == "png_base64"


@pytest.mark.parametrize("change", ["spec", "dimensions", "missing_original", "unknown_kind"])
def test_malformed_preprocessing_markers_rejected(change):
    result = prepare_wire_payload("/encode", payload(), metadata(), image_transport="auto", preprocess_images=True)
    image = result["observations"][0]["images"]["up"]
    if change == "spec":
        image["preprocessing"]["spec"]["longest_edge"] = 128
    elif change == "dimensions":
        image["preprocessing"]["original_size"] = [640, 400]
    elif change == "missing_original":
        del image["preprocessing"]["original_size"]
    else:
        image["preprocessing"]["spec"]["kind"] = "some_other_transform"
    with pytest.raises(ValueError):
        validate_preprocessing_marker(image, decode_image(image), metadata()["image_preprocessing"])


def test_non_rgb_pre_resize_cannot_change_the_model_rgb_only_contract():
    stream = BytesIO()
    Image.new("L", (640, 480)).save(stream, format="PNG")
    value = payload()
    value["observations"][0]["images"]["up"] = {"encoding": "png_bytes", "data": stream.getvalue()}
    with pytest.raises(ValueError, match="RGB"):
        prepare_wire_payload("/encode", value, metadata(), image_transport="auto", preprocess_images=True)


def test_unknown_preprocessing_fails_before_applying_an_unagreed_transform():
    meta = metadata()
    meta["image_preprocessing"]["kind"] = "unrecognized_resize"
    with pytest.raises(ValueError, match="Unsupported"):
        prepare_wire_payload("/encode", payload(), meta, preprocess_images=True)


def test_parallel_image_preparation_preserves_bytes_batch_order_and_caller_data():
    value = {"observations": [payload(size)["observations"][0]
                              for size in [(640, 480), (319, 231), (480, 640), (71, 71)]],
             "seeds": [4, 1, 8, 5]}
    original = deepcopy(value)
    for index, observation in enumerate(value["observations"]):
        observation["state"] = [index]
    kwargs = {"image_transport": "auto", "preprocess_images": True}
    serial = prepare_wire_payload("/encode", value, metadata(), **kwargs)
    parallel = prepare_wire_payload("/encode", value, metadata(), image_workers=4, **kwargs)
    assert serial == parallel
    assert [row["state"][0] for row in parallel["observations"]] == [0, 1, 2, 3]
    assert parallel["seeds"] == original["seeds"]
    assert all(row["images"]["up"]["encoding"] == "png_base64" for row in value["observations"])


def test_producer_resize_matches_existing_client_pixels_without_mutating_native_rgb():
    original = np.random.default_rng(17).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    before = original.copy()
    encoded = encode_preprocessed_image(original, walloss_resize_spec(256))
    assert decode_image(encoded).size == (256, 192)
    assert encoded["preprocessing"]["original_size"] == [640, 480]
    np.testing.assert_array_equal(original, before)
    expected = Image.fromarray(original).resize((256, 192))
    np.testing.assert_array_equal(decode_image(encoded), expected)
    producer = {"observations": [{"images": {"up": encoded}}]}
    client = prepare_wire_payload("/encode", producer, metadata(), image_transport="auto", preprocess_images=True)
    assert client["observations"][0]["images"]["up"]["data"] == base64.b64decode(encoded["data"])


def test_presized_forwarding_only_inspects_header_server_keeps_full_validation(monkeypatch):
    raw = np.zeros((480, 640, 3), dtype=np.uint8)
    encoded = encode_preprocessed_image(raw, walloss_resize_spec(256), encoding="png_bytes")
    # Deliberately corrupt the compressed stream while retaining a valid PNG
    # header. The cheap forwarding path may pass it; authoritative service
    # decode must reject it before a GPU batch or cache insertion.
    corrupt = bytearray(encoded["data"])
    corrupt[corrupt.index(b"IDAT") + 4] ^= 255
    encoded["data"] = bytes(corrupt)
    def unexpected(*args, **kwargs):
        pytest.fail("Forwarding a producer-resized PNG must not fully decode it")
    monkeypatch.setattr("pressb.runtime.image_transport.decode_image", unexpected)
    from test_rl_nodes_inference import Backend, observation
    from pressb.runtime.inference import InferenceService
    backend = Backend()
    service = InferenceService(backend, batch_size=1)
    service.metadata["image_preprocessing"] = walloss_resize_spec(256)
    good = observation()
    good["images"]["up"] = encode_preprocessed_image(raw, walloss_resize_spec(256), encoding="png_bytes")
    bad = observation()
    bad["images"]["up"] = encoded
    forwarded = prepare_wire_payload("/encode", {"observations": [good, bad], "seeds": [1, 2]},
                                     metadata(), image_transport="auto", preprocess_images=True)
    with pytest.raises(ValueError, match="PNG"):
        service.encode(forwarded)
    assert backend.calls == []
    assert service.health()["cache"]["size"] == 0


def test_static_preprocessing_config_validation_and_no_aliasing():
    assert validate_image_preprocessing(None) is None
    spec = walloss_resize_spec(256)
    saved = validate_image_preprocessing(spec)
    saved["longest_edge"] = 128
    assert spec["longest_edge"] == 256
    with pytest.raises(ValueError):
        validate_image_preprocessing({**spec, "kind": "unrecognized"})


@pytest.mark.parametrize("consumer", [None, walloss_resize_spec(128), {"kind": "different"}])
def test_producer_consumer_resize_mismatch_rejected_before_reset(tmp_path, consumer):
    from test_rl_nodes_trainer import make_runner
    runner, inference, simulation = make_runner(tmp_path / "run")
    simulation.meta["image_preprocessing"] = walloss_resize_spec(256)
    if consumer is not None:
        inference.meta["image_preprocessing"] = consumer
    with pytest.raises(ValueError, match="image_preprocessing"):
        runner.run()
    assert simulation.resets == [] and simulation.steps == []
    assert inference.encode_calls == []


def test_resize_contract_legacy_null_and_matching_specs():
    from test_rl_nodes_contract_simulation import metadata as contract_metadata
    from pressb.runtime.contract import negotiate
    inference, simulation = contract_metadata("inference"), contract_metadata()
    prior = negotiate(inference, simulation, "dsrl")
    simulation["image_preprocessing"] = None
    assert negotiate(inference, simulation, "dsrl")["observation"] == prior["observation"]
    inference["image_preprocessing"] = simulation["image_preprocessing"] = walloss_resize_spec(256)
    agreed = negotiate(inference, simulation, "dsrl")
    assert agreed["identity"]["simulation"]["image_preprocessing"] == walloss_resize_spec(256)
    assert agreed["fingerprint"] != prior["fingerprint"]
