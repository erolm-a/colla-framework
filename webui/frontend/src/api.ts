//const API_URL = "http://knowledge-glue-webui-jeffstudentsproject.ida.dcs.gla.ac.uk/api";
const API_URL = "/api"

type POS = "noun" | "verb" | "adjective" | "pronoun" | "adverb"; // ...

export interface ExpressionDefinition {
    pos: POS;
    definition: string;
    examples: string[];
    related: string[];
}

export type ExpressionDefinitionIntent = {"senses": ExpressionDefinition[]};

export interface NewDialogueIntent {
    // Sessions are created with a cookie
    welcomeText: string;
}

export interface InconclusiveIntent {
    response: string;
}

export type Intent = NewDialogueIntent | ExpressionDefinitionIntent | InconclusiveIntent;

export type IntentResponse = {intentType: "newDialogue" | "expressionDefinition" | "inconclusive" } & Intent;


/**
 * Entry point for asking a question (without context)
 * 
 * @param query A question to ask (in natural English)
 * @returns A promise of an IntentResponse.
 */
export async function ask(query: string): Promise<IntentResponse>
{
    const definitionURI = API_URL + "/query/" + query;
    const response = await fetch(definitionURI);

    if (response.ok) {
        return await response.json() as IntentResponse;
    }
    else {
        console.warn(response.statusText);
        return Promise.resolve({intentType: "inconclusive", response: "No"} as IntentResponse);
    }
}

export async function establishDialogue(): Promise<NewDialogueIntent>
{
    const sessionURI = API_URL + "/dialogue/session";
    const response = await fetch(sessionURI);

    if (response.ok) {
        return await response.json() as NewDialogueIntent;
    }
    else
    {
        console.warn(response.statusText);
        return Promise.reject();
        // return Promise.resolve({intentType: "inconclusive", response: "No"} as IntentResponse);
    }
}