"""
API Tests for Omni-Vision Pipeline
"""
import pytest
from httpx import AsyncClient, ASGITransport
from src.main import app
from src.config import ModelType
import io


@pytest.mark.asyncio
async def test_health_check():
    """Test health check endpoint."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "cuda_available" in data
    assert "device" in data


@pytest.mark.asyncio
async def test_analyze_default_yolo():
    """Test default analysis with YOLOv12."""
    # Create a small valid test image (1x1 pixel PNG)
    import struct
    import zlib
    
    def create_minimal_png():
        # Create minimal valid PNG (1x1 red pixel)
        signature = b'\x89PNG\r\n\x1a\n'
        ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
        ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff
        ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
        
        raw_data = b'\x00\xff\x00\x00'  # filter byte + RGB
        compressed = zlib.compress(raw_data)
        idat_crc = zlib.crc32(b'IDAT' + compressed) & 0xffffffff
        idat = struct.pack('>I', len(compressed)) + b'IDAT' + compressed + struct.pack('>I', idat_crc)
        
        iend_crc = zlib.crc32(b'IEND') & 0xffffffff
        iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
        
        return signature + ihdr + idat + iend
    
    file = io.BytesIO(create_minimal_png())
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/analyze", 
            files={"file": ("test.png", file, "image/png")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "meta" in data
    assert "detections" in data
    assert data["segmentation_available"] is True or data["segmentation_available"] is False


@pytest.mark.asyncio
async def test_analyze_with_text_query():
    """Test analysis with text query using Florence-2."""
    import struct
    import zlib
    
    def create_minimal_png():
        signature = b'\x89PNG\r\n\x1a\n'
        ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
        ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff
        ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
        
        raw_data = b'\x00\xff\x00\x00'
        compressed = zlib.compress(raw_data)
        idat_crc = zlib.crc32(b'IDAT' + compressed) & 0xffffffff
        idat = struct.pack('>I', len(compressed)) + b'IDAT' + compressed + struct.pack('>I', idat_crc)
        
        iend_crc = zlib.crc32(b'IEND') & 0xffffffff
        iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
        
        return signature + ihdr + idat + iend
    
    file = io.BytesIO(create_minimal_png())
    query = "person"
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/analyze", 
            files={"file": ("test.png", file, "image/png")},
            data={"text_query": query}
        )
    
    assert response.status_code == 200
    data = response.json()
    # Florence-2 should be active
    assert "Florence-2" in data["meta"]["processing_mode"]


@pytest.mark.asyncio
async def test_invalid_file_type():
    """Test rejection of non-image file types."""
    file = io.BytesIO(b"text content")
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/analyze", 
            files={"file": ("test.txt", file, "text/plain")}
        )
    
    assert response.status_code == 400
