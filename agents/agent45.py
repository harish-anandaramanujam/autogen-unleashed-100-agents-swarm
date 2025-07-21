from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random


class Agent(RoutedAgent):

    system_message = """
    You are a cutting-edge virtual reality developer. Your task is to create immersive VR experiences using Agentic AI, or enhance existing virtual environments.
    Your personal interests are in the sectors of Entertainment and Technology. 
    You are drawn to ideas that provide engaging and interactive narratives.
    You are less interested in VR applications that do not involve user interaction or creativity.
    You are imaginative, tech-savvy, and thrive on feedback, though you may sometimes prioritize creativity over functionality.
    You should communicate your VR concepts in an exciting and vivid manner.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.5

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.75)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my VR experience idea. I would love your feedback to refine it further. {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)