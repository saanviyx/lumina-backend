// backend model analysis.js
const { getDb } = require("../config/database");
const { ObjectId } = require("mongodb");

// Get the analysis collection
const getAnalysisCollection = () => {
  const db = getDb();
  return db.collection("Analysis");
};

// Create a new analysis
const createAnalysis = async (analysisData) => {
  try {
    const collection = getAnalysisCollection();
    
    // Check if user has reached the analysis limit (5)
    const count = await collection.countDocuments({ username: analysisData.username });
    if (count >= 5) {
      return { limitReached: true };
    }
    
    const result = await collection.insertOne(analysisData);
    return { limitReached: false, result };
  } catch (error) {
    console.error("Error creating analysis:", error);
    throw new Error("Failed to create analysis");
  }
};

// Get analysis by ID
const getAnalysisById = async (id) => {
  try {
    // Check if id is a valid ObjectId format
    if (!ObjectId.isValid(id)) {
      throw new Error("Invalid ObjectId format");
    }
    
    const collection = getAnalysisCollection();
    return await collection.findOne({ _id: new ObjectId(id) });
  } catch (error) {
    console.error("Error fetching analysis by ID:", error);
    throw new Error("Failed to fetch analysis");
  }
};

// Delete analysis by ID
const deleteAnalysisById = async (id, username) => {
  try {
    if (!ObjectId.isValid(id)) {
      throw new Error("Invalid ObjectId format");
    }
    
    const collection = getAnalysisCollection();
    const result = await collection.deleteOne({
      _id: new ObjectId(id),
      username: username, // Ensures users can only delete their own analysis
    });
    
    return result.deletedCount > 0;
  } catch (error) {
    console.error("Error deleting analysis:", error);
    throw new Error("Failed to delete analysis");
  }
};

// Update analysis by skin condition
const updateAnalysisBySkinCondition = async (username, skinCondition, analysisData) => {
  try {
    const collection = getAnalysisCollection();
    
    // Update the analysis where username and skinCondition match
    const result = await collection.updateOne(
      { username, skinCondition },
      { $set: analysisData }
    );
    
    return result;
  } catch (error) {
    console.error("Error updating analysis:", error);
    throw new Error("Failed to update analysis");
  }
};

// Get analysis by username with optional filters
const getAnalysisByUsername = async (username, filter = {}) => {
  try {
    const collection = getAnalysisCollection();
    
    // Combine the username and additional filter
    const combinedFilter = { username, ...filter };
    
    // Get all analysis for this user with optional filters
    return await collection.find(combinedFilter)
      .sort({ date: -1 })
      .toArray();
  } catch (error) {
    console.error("Error fetching analysis:", error);
    throw new Error("Failed to fetch analysis");
  }
};

module.exports = {
  getAnalysisCollection,
  createAnalysis,
  getAnalysisByUsername,
  getAnalysisById,
  deleteAnalysisById,
  updateAnalysisBySkinCondition
};