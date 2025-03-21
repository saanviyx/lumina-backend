const express = require('express');
const router = express.Router();
const { runMLModel } = require('../controllers/ml');
const authenticateToken = require('../middleware/auth');
const { getUsersCollection } = require('../models/user');

// Route to analyze skin and get recommendations
router.post('/analyze', async (req, res) => {
  try {
    const { imageBase64} = req.body;
    
    // Validate input
    if (!imageBase64) {
      return res.status(400).json({ message: 'No image provided' });
    }
    
    // Run the ML model
    const result = await runMLModel(imageBase64 || '');
    
    // Return the result
    res.status(200).json({ 
      success: true, 
      routine: result 
    });
    
  } catch (error) {
    console.error('Error analyzing image:', error);
    res.status(500).json({ 
      success: false, 
      message: 'Error analyzing image', 
      error: error.message 
    });
  }
});

// Route to save the routine to the user's profile
router.post('/save-routine', authenticateToken, async (req, res) => {
  try {
    const { imageBase64, routine } = req.body;
    const username = req.user.username;
    
    // Validate input
    if (!routine) {
      return res.status(400).json({ message: 'No routine provided' });
    }
    
    const usersCollection = getUsersCollection();
    
    // Create analysis data object
    const analysisData = { 
      id: Date.now().toString(), 
      date: new Date(), 
      imageBase64, 
      routine 
    };
    
    // Update user document
    const result = await usersCollection.updateOne(
      { username }, 
      { 
        $inc: { analysisCount: 1 }, 
        $push: { 
          analysis: analysisData, 
          routines: routine 
        } 
      }
    );
    
    if (result.modifiedCount === 0) {
      return res.status(404).json({ message: 'User not found' });
    }
    
    // Get updated user
    const updatedUser = await usersCollection.findOne({ username });
    
    res.status(200).json({ 
      message: 'Routine saved successfully', 
      analysisCount: updatedUser.analysisCount, 
      analysis: updatedUser.analysis 
    });
    
  } catch (error) {
    console.error('Error saving routine:', error);
    res.status(500).json({ 
      message: 'Server error while saving routine', 
      error: error.message 
    });
  }
});

module.exports = router;