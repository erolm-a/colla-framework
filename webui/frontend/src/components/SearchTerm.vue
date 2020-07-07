<template>
    <div>
        <b-form @submit="onSubmit">
            <b-form-input
                id="term-input"
                v-model="searchTerm"
                required
                placeholder="Enter the term to search">
            </b-form-input>
        </b-form>
    </div>
</template>

<script lang="ts">
import { Component, Prop, Vue} from 'vue-property-decorator';

@Component
export default class SearchTerm extends Vue {
    searchTerm: string = ""

    onSubmit(event): void {
        console.log(`Searching for: ${this.searchTerm}`)
    }
}

