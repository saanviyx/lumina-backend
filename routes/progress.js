const express = require("express");
const authenticateToken = require("../middleware/auth");
const { getUsersCollection } = require("../models/user");
const {
   getUserCalendarEntries,
   markDayCompleted,
   markDayUncompleted,
   getStreakStats,
   initializeUserCalendar
} = require("../models/calendar");

const router = express.Router();

// Get user progress data
router.get("/", authenticateToken, async (req, res) => {
  try {
    const username = req.user.username;
    const usersCollection = getUsersCollection();
    const user = await usersCollection.findOne({ username }, { projection: { password: 0 } });
    
    if (!user) return res.status(404).json({ message: "User not found" });
    
    // Initialize calendar for new users
    await initializeUserCalendar(username);
    
    // Get calendar entries
    const calendarEntries = await getUserCalendarEntries(username);
    
    // Get stats
    const stats = await getStreakStats(username);
    
    res.status(200).json({
      analysis: user.analysis || [],
      routines: user.routines || [],
      analysisCount: user.analysisCount || 0,
      calendar: {
        entries: calendarEntries,
        stats: stats
      }
    });
  } catch (error) {
    console.error("Progress retrieval error:", error);
    res.status(500).json({ message: "Server error while retrieving progress" });
  }
});

// Mark a day as completed
router.post("/complete", authenticateToken, async (req, res) => {
  try {
    const { date } = req.body;
    const username = req.user.username;
    
    if (!date) {
      return res.status(400).json({ message: "Date is required" });
    }
    
    await markDayCompleted(username, date);
    
    // Get updated stats
    const stats = await getStreakStats(username);
    
    res.status(200).json({
      message: "Day marked as completed",
      stats: stats
    });
  } catch (error) {
    console.error("Error marking day as completed:", error);
    res.status(500).json({ message: "Server error" });
  }
});

// Mark a day as not completed
router.post("/uncomplete", authenticateToken, async (req, res) => {
  try {
    const { date } = req.body;
    const username = req.user.username;
    
    if (!date) {
      return res.status(400).json({ message: "Date is required" });
    }
    
    await markDayUncompleted(username, date);
    
    // Get updated stats
    const stats = await getStreakStats(username);
    
    res.status(200).json({
      message: "Day marked as not completed",
      stats: stats
    });
  } catch (error) {
    console.error("Error marking day as not completed:", error);
    res.status(500).json({ message: "Server error" });
  }
});

module.exports = router;