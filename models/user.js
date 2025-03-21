// Updated user.js model to remove unnecessary arrays

const { getDb } = require("../config/database");

// Get the users collection
function getUsersCollection() {
  return getDb().collection("Users");
}

// Create a new user - Fixed to not have empty arrays
async function createUser(userData) {
  const usersCollection = getUsersCollection();
  // Add default fields without analysis and routines arrays
  const newUser = {
    ...userData,
    createdAt: new Date(),
    analysisCount: 0
  };
  
  try {
    const result = await usersCollection.insertOne(newUser);
    return { success: true, userId: result.insertedId };
  } catch (error) {
    if (error.code === 11000) {
      // Duplicate key error
      return { success: false, error: 'Username or email already exists' };
    }
    throw error;
  }
}

// Find a user by username
async function findUserByUsername(username) {
  const usersCollection = getUsersCollection();
  return await usersCollection.findOne({ username });
}

// Find a user by email
async function findUserByEmail(email) {
  const usersCollection = getUsersCollection();
  return await usersCollection.findOne({ email });
}

// Update a user
async function updateUser(username, updateData) {
  const usersCollection = getUsersCollection();
  const result = await usersCollection.updateOne(
    { username },
    { $set: updateData }
  );
  return result.modifiedCount > 0;
}

// Decrement analysis count - New function
async function decrementAnalysisCount(username) {
  const usersCollection = getUsersCollection();
  const result = await usersCollection.updateOne(
    { username, analysisCount: { $gt: 0 } },
    { $inc: { analysisCount: -1 } }
  );
  return result.modifiedCount > 0;
}

module.exports = {
  getUsersCollection,
  createUser,
  findUserByUsername,
  findUserByEmail,
  updateUser,
  decrementAnalysisCount
};