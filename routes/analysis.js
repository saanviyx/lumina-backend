//backend routes analysis.js
const express = require("express");
const authenticateToken = require("../middleware/auth");
const { getUsersCollection, decrementAnalysisCount } = require("../models/user");
const { 
  createAnalysis, 
  getAnalysisByUsername, 
  getAnalysisById,
  deleteAnalysisById,
  updateAnalysisBySkinCondition,
} = require("../models/analysis");

const router = express.Router();

// Save analysis with selected budget range
router.post("/save", authenticateToken, async (req, res) => {
  try {
    const { routine, selectedBudget, skinCondition, confidence } = req.body;
    const username = req.user.username;

    // Create analysis object
    const analysisData = { 
      username,
      date: new Date(),
      routine,
      skinCondition,
      confidence,
      selectedBudget,
      createdAt: new Date()
    };

    // Check if analysis with the same skin condition already exists
    const existingAnalysis = await getAnalysisByUsername(username, { skinCondition });
    
    let result;
    
    if (existingAnalysis && existingAnalysis.length > 0) {
      // Update existing analysis instead of creating a new one
      result = await updateAnalysisBySkinCondition(username, skinCondition, analysisData);
      
      res.status(200).json({ 
        message: "Analysis updated successfully", 
        analysisId: existingAnalysis[0]._id,
        updated: true
      });
    } else {
      // Check if user has reached the limit
      const usersCollection = getUsersCollection();
      const user = await usersCollection.findOne(
        { username },
        { projection: { analysisCount: 1 } }
      );
      
      if (user && user.analysisCount >= 5) {
        return res.status(400).json({ 
          message: "You have reached the limit of 5 types of analysis. Please delete an existing analysis first."
        });
      }
      
      // Save to the Analysis collection as a new entry
      result = await createAnalysis(analysisData);
      
      // Update the user's analysis count in the users collection
      await usersCollection.updateOne(
        { username }, 
        { $inc: { analysisCount: 1 } }
      );
      
      res.status(200).json({ 
        message: "Analysis saved successfully", 
        analysisId: result.insertedId,
        updated: false
      });
    }
  } catch (error) {
    console.error("Analysis save error:", error);
    res.status(500).json({ message: "Server error while saving analysis" });
  }
});

// Get analysis history for user
router.get("/history", authenticateToken, async (req, res) => {
  try {
    const username = req.user.username;
    
    // Get analysis from the collection
    const analysis = await getAnalysisByUsername(username);
    
    // Get user analysis count
    const usersCollection = getUsersCollection();
    const user = await usersCollection.findOne(
      { username },
      { projection: { analysisCount: 1 } }
    );
    
    if (!user) {
      return res.status(404).json({ message: "User not found" });
    }
    
    res.status(200).json({
      username,
      analysisCount: user.analysisCount || 0,
      analysis: analysis || []
    });
  } catch (error) {
    console.error("Error fetching analysis history:", error);
    res.status(500).json({ message: "Server error while fetching analysis history" });
  }
});

// Get user's analysis count
router.get("/count", authenticateToken, async (req, res) => {
  try {
    const username = req.user.username;
    
    // Get user from the collection to check their analysis count
    const usersCollection = getUsersCollection();
    const user = await usersCollection.findOne(
      { username },
      { projection: { analysisCount: 1 } }
    );
    
    if (!user) {
      return res.status(404).json({ message: "User not found" });
    }
    
    // Return the count
    res.status(200).json({
      count: user.analysisCount || 0
    });
  } catch (error) {
    console.error("Error fetching analysis count:", error);
    res.status(500).json({ message: "Server error while fetching analysis count" });
  }
});

// Get specific analysis by ID
router.get("/:analysisId", authenticateToken, async (req, res) => {
  try {
    const { analysisId } = req.params;
    const username = req.user.username;
    
    // Get analysis from the collection
    const analysis = await getAnalysisById(analysisId);
    
    if (!analysis || analysis.username !== username) {
      return res.status(404).json({ message: "Analysis not found" });
    }
    
    res.status(200).json({
      analysis
    });
  } catch (error) {
    console.error("Error fetching analysis:", error);
    res.status(500).json({ message: "Server error while fetching analysis" });
  }
});

// New endpoint to delete an analysis
router.delete("/:analysisId", authenticateToken, async (req, res) => {
  try {
    const { analysisId } = req.params;
    const username = req.user.username;
    
    // Delete the analysis
    const deleted = await deleteAnalysisById(analysisId, username);
    
    if (!deleted) {
      return res.status(404).json({ message: "Analysis not found or not authorized to delete" });
    }
    
    // Decrement the user's analysis count
    await decrementAnalysisCount(username);
    
    res.status(200).json({ message: "Analysis deleted successfully" });
  } catch (error) {
    console.error("Error deleting analysis:", error);
    res.status(500).json({ message: "Server error while deleting analysis" });
  }
});

module.exports = router;