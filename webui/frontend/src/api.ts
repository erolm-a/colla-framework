//const API_URL = "http://knowledge-glue-webui-jeffstudentsproject.ida.dcs.gla.ac.uk/api";
const API_URL = "/api"

type POS = "noun" | "verb" | "adjective" | "pronoun" | "adverb"; // ...

export interface ExpressionDefinition {
    pos: POS;
    definition: string;
    example: string;
    related: string[];
}

export type ExpressionDefinitionIntent = {"senses": ExpressionDefinition[]};

export interface InconclusiveIntent {
    response: string;
}

export type Intent = ExpressionDefinitionIntent | InconclusiveIntent;

export type IntentResponse = {intentType: "expressionDefinition" | "inconclusional" } & Intent;


/**
 * Entry point for asking a question (without context)
 * 
 * @param query A question to ask (in natural English)
 * @returns A promise of an IntentResponse.
 */
export async function ask(query: string): Promise<IntentResponse>
{
    const definitionApi = API_URL + "/query/" + query;
    const response = await fetch(definitionApi);

    if (response.ok) {
        return await response.json() as IntentResponse;
    }
    else {
        console.warn(response.statusText);
        return Promise.resolve({intentType: "inconclusional", response: "No"} as IntentResponse);
    }
}
