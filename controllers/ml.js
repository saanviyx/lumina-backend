const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

/**
 * Runs the ML model on an image and returns the analysis results
 * @param {string} imageBase64 - Base64 encoded image data
 * @returns {Promise<string>} - The analysis results
 */

function runMLModel(imageBase64) {
  return new Promise((resolve, reject) => {
    // Create a temporary directory to store the image
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'skinanalysis-'));
    const imagePath = path.join(tempDir, 'image.jpg');
    
    // Remove the base64 header (e.g., "data:image/jpeg;base64,")
    const base64Data = imageBase64.replace(/^data:image\/\w+;base64,/, '');
    
    // Write the image to a temporary file
    fs.writeFileSync(imagePath, Buffer.from(base64Data, 'base64'));
    
    // Get the absolute path to the ML script
    const mlScriptPath = path.resolve(__dirname, '../ml/scripts/main.py');
    
    // Run the ML script using Python
    const pythonProcess = spawn('python', [mlScriptPath, imagePath], {
        env: { ...process.env, PYTHONIOENCODING: 'utf-8' }
      });
    
    let result = '';
    let error = '';
    
    // Collect stdout data
    pythonProcess.stdout.on('data', (data) => {
      result += data.toString();
    });
    
    // Collect stderr data
    pythonProcess.stderr.on('data', (data) => {
      error += data.toString();
    });
    
    // Handle process completion
    pythonProcess.on('close', (code) => {
      // Clean up the temporary file
      try {
        fs.unlinkSync(imagePath);
        fs.rmdirSync(tempDir);
      } catch (cleanupError) {
        console.error('Error cleaning up temporary files:', cleanupError);
      }
      
      if (code === 0) {
        // Process completed successfully
        resolve(result);
      } else {
        // Process failed
        reject(new Error(`ML process failed with code ${code}. Error: ${error}`));
      }
    });
  });
}

module.exports = { runMLModel };