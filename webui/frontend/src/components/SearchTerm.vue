<template>
    <div>
        <b-form @submit="onSubmit" inline class="d-flex mb-2">
            <b-form-input
                id="term-input"
                v-model="keyword"
                class="mb-2 mr-sm-2 mb-sm-0 flex-fill"
                required
                placeholder="Enter the term to search">
            </b-form-input>
            <b-button variant="primary" disabled v-if="searching">
                <b-spinner small type="grow" />
                Loading...
            </b-button>
            <b-button variant="primary" type="submit" v-else>
                Search
            </b-button>
        </b-form>

        <b-alert show="errors.length > 0" variant="danger">
            {{error}}
        </b-alert>
        <ul class="list-group">
            <li class="list-group-item" v-for="result in results" :key="result">
                <!-- {{result}} -->
                <i>({{result.pos}})</i> {{result.definition}}
            </li>
        </ul>
    </div>
</template>

<script lang="ts">

import Vue from 'vue';
import {ask, ExpressionDefinition, ExpressionDefinitionIntent, InconclusiveIntent } from '../api';

export default Vue.extend({
    name: "SearchTerm",

    data() {
        return {
            keyword: "",
            results: [] as ExpressionDefinition[],
            searching: false,
            error: "",
            showErrorAlert: false
        }
    },

    methods: {
        async onSubmit(event: Event) {
            event.preventDefault();
            this.searching = true;
            try {
                const response = await ask(this.keyword);
                console.log(response);
                if (response.intentType == "expressionDefinition")
                {
                    this.results = (response as ExpressionDefinitionIntent).senses;
                }
                else
                {
                    console.log("Not what I expected");
                    this.error = (response as InconclusiveIntent).response;
                    this.showErrorAlert = true;
                }
            }
            catch (error)
            {
                console.warn(error);
            }
            this.searching = false;
        }
    }
});
</script>
