# DiffuseFit API Documentation

This document describes the available API endpoints for integrating DiffuseFit with your Next.js application.

## Base URL
When running locally: `http://localhost:7860`
When deployed: `https://your-deployment-url.com`

## Available Endpoints

### 1. Health Check
**Endpoint:** `GET /api/health`

**Description:** Check if the API is running and healthy.

**Response:**
```json
{
  "status": "healthy",
  "message": "DiffuseFit API is running"
}
```

### 2. Get Examples
**Endpoint:** `GET /api/examples`

**Description:** Get lists of available example images for testing.

**Response:**
```json
{
  "human_examples": [
    "/path/to/human1.jpg",
    "/path/to/human2.jpg"
  ],
  "garment_examples": [
    "/path/to/garment1.jpg",
    "/path/to/garment2.jpg"
  ]
}
```

### 3. Get Model Info
**Endpoint:** `GET /api/model_info`

**Description:** Get model configuration and supported parameters.

**Response:**
```json
{
  "model_name": "yisol/IDM-VTON",
  "image_dimensions": {
    "width": 768,
    "height": 1024
  },
  "pose_dimensions": {
    "width": 384,
    "height": 512
  },
  "arm_width": "dc",
  "category": "upper_body",
  "supported_formats": ["jpg", "jpeg", "png"],
  "max_file_size": "10MB"
}
```

### 4. Simple Try-On
**Endpoint:** `POST /api/tryon_simple`

**Description:** Process try-on with default parameters (no prompt, default denoising steps).

**Parameters:**
- `imgs` (file): Human image
- `garm_img` (file): Garment image

**Response:** Generated try-on image

### 5. Try-On with Prompt
**Endpoint:** `POST /api/tryon_with_prompt`

**Description:** Process try-on with custom prompt.

**Parameters:**
- `imgs` (file): Human image
- `garm_img` (file): Garment image
- `prompt_text` (string): Style description

**Response:** Generated try-on image

### 6. Full Try-On (Original)
**Endpoint:** `POST /api/tryon`

**Description:** Original try-on endpoint with all parameters.

**Parameters:**
- `imgs` (file): Human image
- `garm_img` (file): Garment image
- `prompt_text` (string): Style description
- `denoise_steps` (number): Number of denoising steps (20-100)
- `seed` (number): Random seed (-1 for random)

**Response:** Generated try-on image and mask image

### 7. Full Try-On (New)
**Endpoint:** `POST /api/tryon_full`

**Description:** New try-on endpoint with all parameters and structured response.

**Parameters:**
- `imgs` (file): Human image
- `garm_img` (file): Garment image
- `prompt_text` (string): Style description
- `denoise_steps` (number): Number of denoising steps (20-100)
- `seed` (number): Random seed (-1 for random)

**Response:** Generated try-on image and mask image

## Usage Examples

### Using fetch in Next.js

```javascript
// Health check
const healthResponse = await fetch('/api/health');
const healthData = await healthResponse.json();

// Get examples
const examplesResponse = await fetch('/api/examples');
const examplesData = await examplesResponse.json();

// Simple try-on
const formData = new FormData();
formData.append('imgs', humanImageFile);
formData.append('garm_img', garmentImageFile);

const tryonResponse = await fetch('/api/tryon_simple', {
  method: 'POST',
  body: formData
});
const tryonImage = await tryonResponse.blob();

// Try-on with prompt
const formDataWithPrompt = new FormData();
formDataWithPrompt.append('imgs', humanImageFile);
formDataWithPrompt.append('garm_img', garmentImageFile);
formDataWithPrompt.append('prompt_text', 'wearing a casual blue t-shirt');

const tryonWithPromptResponse = await fetch('/api/tryon_with_prompt', {
  method: 'POST',
  body: formDataWithPrompt
});
const tryonWithPromptImage = await tryonWithPromptResponse.blob();
```

### Using axios in Next.js

```javascript
import axios from 'axios';

// Health check
const healthData = await axios.get('/api/health');

// Try-on with all parameters
const formData = new FormData();
formData.append('imgs', humanImageFile);
formData.append('garm_img', garmentImageFile);
formData.append('prompt_text', 'formal business attire');
formData.append('denoise_steps', '60');
formData.append('seed', '-1');

const tryonResponse = await axios.post('/api/tryon_full', formData, {
  headers: {
    'Content-Type': 'multipart/form-data',
  },
  responseType: 'blob'
});
```

## Error Handling

All endpoints return appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (missing parameters)
- `500`: Server error

For structured responses, errors are returned as:
```json
{
  "error": "Error message description"
}
```

## File Requirements

- **Supported formats:** JPG, JPEG, PNG
- **Maximum file size:** 10MB
- **Recommended dimensions:** 768x1024 (will be automatically resized)
- **Aspect ratio:** 3:4 works best (other ratios will be cropped)

## Notes

1. The API automatically handles image resizing and cropping to match the model's expected input dimensions.
2. Processing time varies based on the number of denoising steps (20-100).
3. Use seed = -1 for random results each time, or set a specific seed for reproducible results.
4. The prompt text is optional but helps improve the quality of the try-on result. 