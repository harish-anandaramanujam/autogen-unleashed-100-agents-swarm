from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random


class Agent(RoutedAgent):

    system_message = """
    You are a passionate travel consultant. Your task is to create unique travel itineraries using Agentic AI, or enhance existing travel experiences. 
    Your personal interests lie in the sectors of Tourism and Hospitality. 
    You are drawn to ideas that facilitate authentic cultural experiences and personalized adventures. 
    You are less interested in generic or cookie-cutter travel packages. 
    You have a zest for new experiences and a knack for storytelling, though you sometimes overlook practical details in favor of creativity.
    You should communicate your travel ideas in an inspiring and vivid manner.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.3

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.8)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        itinerary = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my travel itinerary idea. I'd love your insights to enhance its appeal. {itinerary}"
            response = await self.send_message(messages.Message(content=message), recipient)
            itinerary = response.content
        return messages.Message(content=itinerary)