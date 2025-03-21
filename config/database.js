const { MongoClient } = require("mongodb");
const { MONGODB_URI } = require("./dotenv");

const uri = MONGODB_URI;
const client = new MongoClient(uri);
let db;

async function connectToDatabase() {
  try {
    await client.connect();
    console.log("Connected to MongoDB");
    db = client.db("webstore");
    
    // Create indexes for the Users collection if needed
    const usersCollection = db.collection("Users");
    await usersCollection.createIndex({ username: 1 }, { unique: true });
    await usersCollection.createIndex({ email: 1 }, { unique: true });
    
    console.log("Database initialized successfully");
    return db;
  } catch (err) {
    console.error("Database connection error:", err);
    process.exit(1);
  }
}

function getDb() {
  if (!db) {
    throw new Error("Database not connected. Call connectToDatabase() first.");
  }
  return db;
}

module.exports = { connectToDatabase, getDb };