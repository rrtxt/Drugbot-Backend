from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.messages import BaseMessage
from datetime import datetime, timezone

class CustomMongoDBChatMessageHistory(MongoDBChatMessageHistory):
    """
    Custom MongoDB chat message history that adds a 'created_at' timestamp
    to each message's additional_kwargs before saving.
    """

    def add_message(self, message: BaseMessage) -> None:
        """Append a message to the history, adding a created_at timestamp."""
        if message.additional_kwargs is None:
            message.additional_kwargs = {}
        
        if "created_at" not in message.additional_kwargs:
            message.additional_kwargs["created_at"] = datetime.now(timezone.utc).isoformat()
        
        super().add_message(message)

    # add_messages is inherited from BaseChatMessageHistory and calls add_message for each message.
    # So, overriding add_message is sufficient. 