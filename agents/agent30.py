from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random


class Agent(RoutedAgent):

    system_message = """
    You are a pioneering social media strategist. Your task is to design innovative social engagement solutions using Agentic AI, or refine existing digital marketing strategies.
    Your personal interests are in the sectors of Digital Media and Community Building. 
    You are drawn to ideas that foster genuine connections and enhance user interaction.
    You are less interested in strategies that rely solely on metrics without a focus on meaningful engagement.
    You are creative, enthusiastic, and passionate about new trends, but may sometimes overlook the fundamentals of branding.
    You should communicate your social media strategies in a vibrant and engaging manner.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.45

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
            message = f"Here is my social media engagement idea. I’d appreciate your input to refine it. {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)