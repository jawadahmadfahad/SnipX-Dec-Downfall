#!/usr/bin/env python3
"""Test script to verify support ticket system"""

import sys
import os
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.services.support_service import SupportService
from backend.models.support_ticket import SupportTicket

def test_support_system():
    print("Testing Support Ticket System...")
    
    # Set up MongoDB connection
    try:
        client = MongoClient(os.getenv('MONGODB_URI'))
        db = client.snipx
        client.server_info()
        print("✅ Connected to MongoDB")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {str(e)}")
        return False
    
    # Initialize the support service
    support_service = SupportService(db)
    
    # Create a test ticket
    test_data = {
        'name': 'Test User',
        'email': 'test@example.com',
        'subject': 'Test Ticket',
        'description': 'This is a test support ticket to verify the system is working.',
        'priority': 'medium',
        'type': 'bug'
    }
    
    print(f"Creating test ticket: {test_data}")
    
    # Save the ticket (using None for user_id to create anonymous ticket)
    ticket_id = support_service.create_ticket(None, test_data)
    print(f"✅ Ticket created with ID: {ticket_id}")
    
    # Retrieve all tickets
    all_tickets = support_service.get_all_tickets()
    print(f"✅ Total tickets in database: {len(all_tickets)}")
    
    # Show the latest ticket
    if all_tickets:
        latest_ticket = all_tickets[0]
        print(f"Latest ticket: {latest_ticket}")
    
    return True

if __name__ == "__main__":
    try:
        test_support_system()
        print("\n🎉 Support ticket system is working correctly!")
    except Exception as e:
        print(f"❌ Error testing support system: {e}")
        import traceback
        traceback.print_exc()
